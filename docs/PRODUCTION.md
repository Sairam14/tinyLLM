# tinyLLM Production Deployment Guide

Complete reference for deploying the German LLM to production with training, serving, and edge inference.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Installation](#installation)
3. [Training](#training)
4. [Inference Optimization](#inference-optimization)
5. [API Serving](#api-serving)
6. [Deployment](#deployment)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Stack Components

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Model** | PyTorch 2.1+ | Training and inference |
| **Tokenizer** | HuggingFace tokenizers | Rust-backed BPE (10-50× faster) |
| **GPU Kernels** | Flash Attention 2 | Automatic via `F.scaled_dot_product_attention` |
| **Training** | PyTorch DDP | Distributed, single/multi-GPU |
| **Precision** | AMP (bfloat16/float16) | Speed + memory efficiency |
| **API** | FastAPI + Uvicorn | REST endpoints with streaming |
| **Deployment** | Docker + CUDA 12.1 | Containerized serving |
| **Inference (CPU)** | llama.cpp + GGUF | Edge/on-premise deployment |
| **Quantization** | ONNX Runtime INT8 | CPU inference optimization |
| **Monitoring** | Prometheus + Grafana | Metrics and dashboards |

### Model Architecture

**Pre-LayerNorm Transformer:**

```
Input → Token Embedding + Positional Encoding
         ↓
    [Transformer Block] × N layers
    ├─ LayerNorm
    ├─ Multi-Head Attention (Flash Attention)
    ├─ Residual Connection
    ├─ LayerNorm
    ├─ FFN (GELU, 4× expansion)
    └─ Residual Connection
         ↓
    Final LayerNorm
         ↓
    Output Projection → Vocabulary Logits
```

**Key improvements over educational `main.py`:**
- Pre-LN instead of Post-LN (training stability without warmup)
- GELU instead of ReLU (empirically better)
- Flash Attention instead of manual SDPA (10-50× faster on V100/A100)
- Weight tying (embedding ↔ lm_head)
- Dropout for regularization
- KV-cache for O(N) inference instead of O(N²)

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU) or CPU-only
- ~5GB disk for model checkpoints
- 16GB+ RAM for training

### Setup

```bash
# Clone repo
git clone https://github.com/your-repo/tinyLLM.git
cd tinyLLM

# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install production dependencies
pip install -e .

# (Optional) Install dev dependencies for testing
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "from tinyllm.model import GermanLM; print('✓ Installation successful')"
```

---

## Training

### Step 1: Prepare Data

Train tokenizer on real German data:

```bash
python scripts/train_tokenizer_hf.py \
    --vocab-size 32000 \
    --max-chars 5000000 \
    --output tokenizer_32k.json
```

This streams from:
- German Wikipedia (~2GB)
- CC-100 German (use 3GB of 65GB total)

**Never** downloads the full corpus, streams on-the-fly.

### Step 2: Configure Training

Create `train_config.json`:

```json
{
    "learning_rate": 3e-4,
    "warmup_steps": 1000,
    "total_steps": 100000,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "max_seq_len": 2048,
    "dtype": "bfloat16",
    "use_gradient_checkpointing": true,
    "checkpoint_every": 500,
    "log_every": 50,
    "use_wandb": true,
    "wandb_project": "tinyllm"
}
```

And `model_config.json`:

```json
{
    "vocab_size": 32000,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 12,
    "max_seq_len": 2048,
    "dropout": 0.1,
    "ffn_mult": 4.0
}
```

### Step 3: Train

```bash
# Single GPU (A100 or V100)
python tinyllm/train.py \
    --model-config model_config.json \
    --train-config train_config.json \
    --tokenizer tokenizer_32k.json

# Multi-GPU (2+ GPUs)
torchrun --nproc_per_node=2 tinyllm/train.py \
    --model-config model_config.json \
    --train-config train_config.json \
    --tokenizer tokenizer_32k.json

# Resume from checkpoint
python tinyllm/train.py \
    --resume checkpoints/step_0500000.pt \
    --tokenizer tokenizer_32k.json
```

**Monitoring:**
- Logs print every 50 steps
- WandB dashboard at https://wandb.ai/ (if enabled)
- Metrics: loss, learning rate, gradient norm, validation loss

**Expected performance:**
- A100 (40GB): ~400 tokens/sec, 10-12 hours for 100k steps
- V100 (32GB): ~150 tokens/sec, 26-30 hours for 100k steps
- With gradient checkpointing: 2-3× larger batches possible

---

## Inference Optimization

### ONNX Export (CPU Inference)

```bash
python tinyllm/export/onnx_export.py \
    --checkpoint checkpoints/step_0100000_final.pt \
    --output-dir exports
```

Outputs:
- `exports/model.onnx` — Full precision (FP32)
- `exports/model_int8.onnx` — Quantized INT8 (3-4× smaller)

**Performance (approximate, on CPU):**
- FP32: 2-5 tokens/sec
- INT8: 3-8 tokens/sec (1.5-2× faster)
- Memory: 512MB (INT8) vs 2GB (FP32)

### GGUF Export (Edge Deployment via llama.cpp)

```bash
# Export to HuggingFace format (compatible with llama.cpp)
python tinyllm/export/gguf_export.py \
    --checkpoint checkpoints/step_0100000_final.pt \
    --output-dir tinyllm_hf

# Convert to GGUF FP16
python llama.cpp/convert_hf_to_gguf.py tinyllm_hf/ \
    --outfile model.gguf \
    --outtype f16

# Quantize to 4-bit
./llama.cpp/llama-quantize model.gguf model_q4.gguf Q4_K_M

# Inference
./llama.cpp/llama-cli -m model_q4.gguf -p "Guten Morgen"
```

**Model sizes:**
- GGUF F16: ~1GB (512×512×12 layers × 2 bytes)
- GGUF Q4_K_M: ~250MB (4× compression)

**Performance (Q4_K_M, on CPU):**
- Apple Silicon M1/M2/M3: 5-10 tokens/sec
- x86 (Ryzen/Xeon): 2-4 tokens/sec

---

## API Serving

### Option 1: FastAPI Development Server

```bash
# Start with default settings
python tinyllm/serving/app.py

# With custom checkpoint/tokenizer
MODEL_CHECKPOINT=checkpoints/final.pt \
TOKENIZER_PATH=tokenizer_32k.json \
python -m uvicorn tinyllm.serving.app:app --reload --port 8000
```

### Option 2: Docker Compose (Recommended)

```bash
# Build and start (FastAPI + Prometheus + Grafana)
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f api

# Stop
docker compose down
```

### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

Response:
```json
{
    "status": "ok",
    "model_loaded": true,
    "device": "cuda"
}
```

**Tokenize:**
```bash
curl -X POST http://localhost:8000/v1/tokenize \
    -H "Authorization: Bearer test-key-1" \
    -H "Content-Type: application/json" \
    -d '{"text": "Guten Morgen"}'
```

**Generate (non-streaming):**
```bash
curl -X POST http://localhost:8000/v1/generate \
    -H "Authorization: Bearer test-key-1" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Das ist",
        "max_new_tokens": 50,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95
    }'
```

Response:
```json
{
    "text": "Das ist ein Test für...",
    "tokens_generated": 42,
    "generation_time_ms": 1234.5
}
```

**Generate (streaming):**
```bash
curl -X POST http://localhost:8000/v1/generate/stream \
    -H "Authorization: Bearer test-key-1" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Das ist", "max_new_tokens": 50}' \
    --no-buffer
```

Server-Sent Events output (one token per line):
```
data: {"delta": "ein", "finish_reason": null}
data: {"delta": " Test", "finish_reason": null}
data: {"delta": "", "finish_reason": "stop"}
```

### Environment Variables

**API Configuration:**
```bash
# API keys (comma-separated)
export API_KEYS="key1,key2,key3"

# Model paths
export MODEL_CHECKPOINT="/path/to/checkpoint.pt"
export TOKENIZER_PATH="/path/to/tokenizer.json"

# Compute
export CUDA_VISIBLE_DEVICES="0"  # GPU ID

# Logging
export WANDB_PROJECT="tinyllm"
export WANDB_ENTITY="your-username"
```

---

## Deployment

### Docker Build

```bash
# Build image
docker build -t tinyllm:latest .

# Test locally
docker run --gpus all -p 8000:8000 tinyllm:latest

# Push to registry
docker tag tinyllm:latest your-registry/tinyllm:latest
docker push your-registry/tinyllm:latest
```

### Kubernetes Deployment

Example `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tinyllm-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: tinyllm-api
  template:
    metadata:
      labels:
        app: tinyllm-api
    spec:
      containers:
      - name: api
        image: your-registry/tinyllm:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1  # 1 GPU per pod
          limits:
            nvidia.com/gpu: 1
        env:
        - name: API_KEYS
          valueFrom:
            secretKeyRef:
              name: tinyllm-secrets
              key: api-keys
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: tinyllm-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: tinyllm-api
```

Deploy:
```bash
kubectl create secret generic tinyllm-secrets --from-literal=api-keys="key1,key2,key3"
kubectl apply -f k8s-deployment.yaml
```

---

## Monitoring

### Prometheus Metrics

Available at `http://localhost:9090/`:

**Key metrics:**
```
llm_requests_total{endpoint="/v1/generate", status="200"}
llm_tokens_generated_total{method="generate"}
llm_generation_latency_seconds (histogram)
llm_model_loaded (0 or 1)
http_request_duration_seconds
```

### Grafana Dashboards

Access at `http://localhost:3000/` (user: admin, password: admin)

**Example dashboard queries:**

Requests per minute:
```
rate(llm_requests_total[1m])
```

Average generation latency:
```
histogram_quantile(0.95, rate(llm_generation_latency_seconds_bucket[5m]))
```

Tokens per second:
```
rate(llm_tokens_generated_total[1m])
```

---

## Performance Benchmarking

Run benchmarks across all inference formats:

```bash
python scripts/benchmark.py
```

Output:
```
Format              | Device | Tokens/sec | Memory (MB) | First-token (ms)
PyTorch FP32        | GPU    | 400        | 2048        | 15
PyTorch BF16 (AMP)  | GPU    | 800        | 1024        | 12
ONNX FP32           | CPU    | 5          | 1024        | 50
ONNX INT8           | CPU    | 8          | 512         | 45
GGUF Q4_K_M         | CPU    | 6          | 256         | 55
```

---

## Troubleshooting

### Training Issues

**CUDA Out of Memory:**
```python
# In train_config.json:
{
    "batch_size": 16,  # Reduce from 32
    "use_gradient_checkpointing": true,  # Enable
    "gradient_accumulation_steps": 2  # Increase
}
```

**Training Loss Not Decreasing:**
- Check data: Are tokens correct?
- Check LR: Try 1e-3, 1e-4, 1e-5
- Check model: Larger model may need more steps
- Check gradients: Are they flowing? Check grad norms in WandB

### Serving Issues

**Port Already in Use:**
```bash
# Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>

# Or use different port
python -m uvicorn tinyllm.serving.app:app --port 8001
```

**Model Not Loading:**
```bash
# Check checkpoint path
ls -lh checkpoints/

# Try dummy model (for testing)
# Set MODEL_CHECKPOINT to non-existent path and it will create a dummy
```

**API Returns 503 (Service Unavailable):**
- Model still loading (wait 30s)
- Checkpoint path incorrect
- GPU out of memory (check `nvidia-smi`)

### Quantization Issues

**ONNX Export Fails:**
```bash
# Make sure ONNX is importable
python -c "import onnx; print(onnx.__version__)"

# Try exporting with simpler config
python tinyllm/export/onnx_export.py --checkpoint checkpoints/step_00000.pt --opset 17
```

**GGUF Conversion Fails:**
```bash
# Ensure llama.cpp is cloned and built
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Verify convert script exists
python convert_hf_to_gguf.py --help
```

---

## Performance Tuning

### For Throughput (many requests)

```python
# train_config.json:
{
    "batch_size": 64,
    "use_gradient_checkpointing": false,  # Trade memory for speed
    "dtype": "bfloat16"  # A100 only
}

# Docker: multiple replicas
# docker compose scale api=4
```

### For Latency (low p99)

```python
# train_config.json:
{
    "batch_size": 8,
    "use_gradient_checkpointing": true,
    "dtype": "float16"  # Slightly faster than bfloat16 on V100
}

# Use KV-cache generation (automatic in KVCacheGenerator)
# Avoid streaming (adds ~50ms SSE overhead)
```

### For Cost (CPU inference)

- Use GGUF Q4_K_M format (~256MB vs 2GB)
- Deploy on CPU nodes (5-10× cheaper than GPU)
- Batch requests when possible
- Use caching layer (Redis) for frequent queries

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Train tokenizer on German data
3. ✅ Train model (or load pretrained checkpoint)
4. ✅ Export for deployment (ONNX, GGUF)
5. ✅ Deploy API (Docker Compose or Kubernetes)
6. ✅ Monitor with Prometheus/Grafana
7. 📊 Optimize performance based on metrics
8. 🚀 Scale horizontally with load balancing

---

## References

- **Flash Attention:** https://github.com/Dao-AILab/flash-attention
- **PyTorch DDP:** https://pytorch.org/docs/stable/notes/ddp.html
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **ONNX Runtime:** https://onnxruntime.ai/
- **HuggingFace Tokenizers:** https://huggingface.co/docs/tokenizers/

---

## Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Email:** sairam.sundaram@gmail.com
