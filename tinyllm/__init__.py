"""
tinyLLM — Production-Grade German Language Model

A complete stack for training, serving, and deploying a tiny transformer-based
language model optimized for German text. Includes:
- Tokenization: HuggingFace tokenizers (Rust-backed BPE)
- Model: Pre-LN transformer with Flash Attention (PyTorch)
- Training: DDP-ready with AMP, gradient checkpointing, and monitoring
- Inference: KV-cache, quantization (ONNX INT8, GGUF)
- Serving: FastAPI with streaming, auth, metrics
- Deployment: Docker, edge (CPU via llama.cpp)
"""

__version__ = "0.1.0"
__author__ = "Sairam Sundaram"

from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.model import GermanLM
from tinyllm.tokenizer import GermanTokenizer

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "GermanLM",
    "GermanTokenizer",
]
