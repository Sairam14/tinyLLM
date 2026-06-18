# tinyLLM Pipeline Guide

**Start here:** Run `python main.py` to see the complete pipeline in action.

## Architecture Overview

```
German Text Input
       ↓
┌──────────────────────────────────────────────────────┐
│ PART 1: TOKENIZATION (BPE)                          │
│                                                      │
│ • Normalize German text (preserve umlauts)          │
│ • Split into words (respects compound words)        │
│ • Byte-Pair Encoding: merge frequent byte pairs     │
│ • Output: token IDs                                 │
│                                                      │
│ File: bpe_tokenizer.py                              │
│ Entry: demo.py                                      │
└──────────────────────────────────────────────────────┘
       ↓ [token_ids]
┌──────────────────────────────────────────────────────┐
│ PART 2: EMBEDDING + POSITIONAL ENCODING             │
│                                                      │
│ • Convert token IDs to embeddings (128-dim)         │
│ • Add positional encoding (sine/cosine patterns)    │
│ • Output: [batch, seq_len, 128]                     │
│                                                      │
│ File: multi_head_attention.py                       │
│ Class: PositionalEncoding                           │
└──────────────────────────────────────────────────────┘
       ↓ [embeddings + positions]
┌──────────────────────────────────────────────────────┐
│ PART 3: MULTI-HEAD ATTENTION LAYERS                 │
│                                                      │
│ Layer 1: Self-Attention (4 heads)                   │
│ ├─ Query, Key, Value projections                    │
│ ├─ Scaled dot-product attention (per head)          │
│ ├─ Concatenate heads → output projection            │
│ └─ Add residual connection + layer norm             │
│                                                      │
│ Feed-Forward Network (expand → ReLU → project)      │
│ └─ Add residual connection + layer norm             │
│                                                      │
│ Layer 2: (repeat)                                   │
│                                                      │
│ File: multi_head_attention.py (core)                │
│ File: mha_examples.py (TransformerBlock)            │
└──────────────────────────────────────────────────────┘
       ↓ [transformed embeddings]
┌──────────────────────────────────────────────────────┐
│ PART 4: OUTPUT PROJECTION                           │
│                                                      │
│ • Project from 128-dim to vocab_size (213)          │
│ • Output: logits for next token prediction          │
│ • Output shape: [batch, seq_len, vocab_size]        │
│                                                      │
│ File: main.py (GermanLanguageModel.lm_head)         │
└──────────────────────────────────────────────────────┘
       ↓ [logits]
Next Token Prediction / Text Generation
```

## File Structure

### Core Components

| File | Purpose | Key Classes |
|------|---------|-------------|
| **bpe_tokenizer.py** | BPE tokenization for German | `BPETokeniser` |
| **multi_head_attention.py** | Core MHA implementation | `PositionalEncoding`, `ScaledDotProductAttention`, `MultiHeadAttention` |
| **mha_examples.py** | Advanced patterns & complete model | `TransformerBlock`, `SimpleLanguageModel` |

### Entry Points

| File | Purpose | What to Run |
|------|---------|-----------|
| **main.py** | 🎯 **START HERE** — Full pipeline demo | `python main.py` |
| **demo.py** | BPE tokenizer demo (Part 1 only) | `python demo.py` |
| **multi_head_attention.py** | MHA core demo (Part 2 only) | `python multi_head_attention.py` |
| **mha_examples.py** | 6 advanced MHA examples | `python mha_examples.py` |

### Documentation

| File | Content |
|------|---------|
| **README.md** | Project overview & setup |
| **PIPELINE.md** | This file — architecture & flow |

## How Everything Connects

### Step 1: Text → Tokens

```python
from bpe_tokenizer import BPETokeniser

tokenizer = BPETokeniser(vocab_size=256)
tokenizer.train(corpus)

text = "Das ist ein Test."
token_ids = tokenizer.encode(text)
# [2, 58, 48, 59, 87, 6, 3]
```

### Step 2: Tokens → Embeddings

```python
import torch
from multi_head_attention import PositionalEncoding

# Convert token IDs to embeddings
token_ids = torch.tensor([[2, 58, 48, 59, 87, 6, 3]])
embeddings = model.token_embedding(token_ids)  # [1, 7, 128]

# Add positional information
pos_encoding = PositionalEncoding(d_model=128)
embedded = pos_encoding(embeddings)  # [1, 7, 128]
```

### Step 3: Embeddings → Attention Layers

```python
from multi_head_attention import MultiHeadAttention

# Multi-head attention (4 heads, 128-dim)
mha = MultiHeadAttention(d_model=128, n_heads=4)

# Self-attention: each token attends to all tokens
output = mha(embedded, embedded, embedded)  # [1, 7, 128]
```

### Step 4: Complete Pipeline

```python
from main import GermanLanguageModel

# Create the complete model
vocab_size = len(tokenizer.token_to_id)  # 213
model = GermanLanguageModel(vocab_size=vocab_size)

# Forward pass
logits = model(token_ids)  # [1, 7, 213]

# Get next token prediction
next_token_id = logits[0, -1, :].argmax().item()
next_token = tokenizer.decode([next_token_id])
```

## Key Concepts

### 1. Tokenization (BPE)

- **Why BPE?** Efficient for German compounds: "Donaudampfschifffahrtsgesellschaft" → 1-3 tokens vs 5+ in English
- **Process:**
  1. Start with characters as tokens
  2. Find most frequent adjacent pair
  3. Merge them into a single token
  4. Repeat until vocab size reached
- **Result:** Smaller vocab, fewer tokens, better efficiency

### 2. Multi-Head Attention

- **Why multiple heads?** Each head learns different representation subspaces
  - Head 1: might learn grammatical structure
  - Head 2: might learn semantic relationships
  - Head 3: might learn syntactic patterns
  - etc.
- **Scaled dot-product:** Prevents attention weights from becoming too sharp/small
- **Causal mask:** Prevents attending to future tokens (autoregressive)

### 3. Positional Encoding

- **Problem:** Transformer doesn't know token order (it's permutation-invariant)
- **Solution:** Add sine/cosine patterns at different frequencies
  - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
- **Why sine/cosine?** Model can learn relative positions (PE(pos + k) is a linear function of PE(pos))

### 4. Residual Connections

```python
# Instead of: x = attn(x)
# We do: x = x + attn(x)

# This helps:
# • Preserve gradient flow (backprop deeper layers)
# • Maintain identity mapping as escape route
# • Train much deeper networks
```

## Model Architecture (from main.py)

```
GermanLanguageModel(
  vocab_size=213,      # German BPE vocabulary
  d_model=128,         # Embedding dimension
  n_heads=4,           # Attention heads per layer
  n_layers=2,          # Transformer blocks
  max_seq_len=256      # Maximum sequence length
)

Layers:
├─ TokenEmbedding: vocab_size → 128
├─ PositionalEncoding: add position info
├─ TransformerBlock 1:
│  ├─ MultiHeadAttention (4 heads, 128-dim)
│  ├─ Residual + LayerNorm
│  ├─ FeedForward (128 → 512 → 128)
│  └─ Residual + LayerNorm
├─ TransformerBlock 2: (same)
└─ LMHead: 128 → vocab_size (213)

Total Parameters: ~451K
```

## Examples

### Example 1: Tokenize German Text

```bash
python demo.py
```

Output:
```
Text: Das ist ein Test.
Token IDs: [2, 58, 48, 59, 87, 6, 3]
Fertility: 1.75 tokens/word
```

### Example 2: See MHA in Action (6 examples)

```bash
python mha_examples.py
```

Examples:
1. Self-attention: Q=K=V (each token attends to all)
2. Cross-attention: Decoder queries encoder
3. Causal attention: Can't look at future tokens
4. Transformer block: MHA + FFN + residuals
5. Simple language model: Complete autoregressive model
6. Attention head analysis: Inspect what each head learns

### Example 3: Full Pipeline

```bash
python main.py
```

Shows:
1. Tokenize German text
2. Forward pass through model
3. Next token predictions
4. Text generation (greedy sampling)

## Next Steps

1. **Understand tokenization:**
   - Read: `bpe_tokenizer.py` (detailed comments)
   - Run: `python demo.py`

2. **Understand attention:**
   - Read: `multi_head_attention.py` (80-line core)
   - Run: `python multi_head_attention.py`
   - Run: `python mha_examples.py` (6 patterns)

3. **See complete system:**
   - Read: `main.py` (unified pipeline)
   - Run: `python main.py`

4. **Extend it:**
   - Add better sampling (temperature, top-k, nucleus)
   - Train on larger German corpus
   - Add beam search decoding
   - Fine-tune on specific German tasks

## Hyperparameters

Key tunable parameters in `main.py`:

```python
# Tokenizer
vocab_size = 256  # BPE vocabulary size

# Model
d_model = 128      # Embedding dimension (larger → more capacity)
n_heads = 4        # Attention heads (more → more parallel learning)
n_layers = 2       # Transformer depth (deeper → more complex patterns)
max_seq_len = 256  # Max input length
```

Recommendations:
- **Tiny:** d_model=64, n_heads=2, n_layers=1
- **Small:** d_model=128, n_heads=4, n_layers=2
- **Medium:** d_model=256, n_heads=8, n_layers=4
- **Large:** d_model=512, n_heads=8, n_layers=12

## Performance Notes

On standard hardware (CPU):

| Operation | Time |
|-----------|------|
| Train BPE tokenizer | 0.5s |
| Forward pass (batch=2, seq_len=10) | 10ms |
| Generate 10 tokens (greedy) | 100ms |

GPU would be 10-50× faster.

## License

MIT — Educational use
