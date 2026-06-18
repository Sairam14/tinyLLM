# Quick Start Guide

## First Time? Run This

```bash
python main.py
```

This will show you the complete pipeline in ~2 seconds:
1. Train a German BPE tokenizer
2. Create a tiny transformer model
3. Process German text through the model
4. Generate predictions

## Next: Understand Each Part

### Part 1: Tokenization (German BPE)
**Time:** 5 minutes

```bash
# Quick demo
python demo.py

# See what's happening
cat bpe_tokenizer.py  # Read the code (100 lines, well-commented)
```

**Key concept:** Break German text into tokens efficiently using Byte-Pair Encoding.

**Why German?** Compound words like "Donaudampfschifffahrtsgesellschaft" become 1-3 tokens instead of 10+ in English.

---

### Part 2: Multi-Head Attention (Transformer Core)
**Time:** 10 minutes

```bash
# See the core mechanism
python multi_head_attention.py

# Understand the implementation (80 lines)
cat multi_head_attention.py

# See 6 practical patterns
python mha_examples.py

# Read detailed explanations
cat PIPELINE.md  # Architecture section
```

**Key concept:** Attention lets each token "look at" other tokens and learn relationships.

**Why multi-head?** Multiple attention heads learn different patterns in parallel:
- One head learns grammar
- Another learns semantics
- Another learns syntax
- etc.

---

### Part 3: Complete System
**Time:** 5 minutes

```bash
# See everything together
cat main.py  # Read the unified pipeline (300 lines)

# Understand the architecture
cat PIPELINE.md  # Full architecture explanation
```

**Key concept:** Connect tokenization + attention to build a language model.

**What it does:**
1. Tokenize German text → [token_ids]
2. Embed tokens + add positions → [embeddings]
3. Process through attention layers → [transformed]
4. Project to vocabulary → [logits for next token]

---

## File Map

```
tinyLLM/
├── main.py                    ← 🎯 START HERE (complete pipeline)
├── QUICK_START.md             ← This file
├── PIPELINE.md                ← Architecture & flow
├── README.md                  ← Project overview
│
├── Part 1: Tokenization
│   ├── bpe_tokenizer.py       ← Core implementation (100 lines)
│   ├── demo.py                ← Quick demo
│   └── train_tokenizer.py     ← Full training on German corpus
│
├── Part 2: Attention
│   ├── multi_head_attention.py ← Core implementation (80 lines)
│   ├── mha_examples.py        ← 6 practical patterns & complete model
│   └── test_mha_german.py     ← Unit tests
│
└── Utilities
    └── .venv/                 ← Virtual environment
```

---

## Common Tasks

### "I want to understand tokenization"
1. Run: `python demo.py`
2. Read: `bpe_tokenizer.py` (top-to-bottom, has step-by-step comments)
3. Read: `train_tokenizer.py` (shows full training on 5MB German corpus)

### "I want to understand attention"
1. Run: `python multi_head_attention.py` (core mechanism)
2. Run: `python mha_examples.py` (6 patterns: self, cross, causal, etc.)
3. Read: `PIPELINE.md` (architecture section)
4. Read: `multi_head_attention.py` (implementation, line-by-line comments)

### "I want to understand the complete system"
1. Run: `python main.py` (see everything working)
2. Read: `PIPELINE.md` (architecture & flow)
3. Read: `main.py` (implementation with detailed comments)

### "I want to modify the model"
Key parameters in `main.py`:

```python
# Tokenizer
vocab_size = 256

# Model architecture
d_model = 128      # Try: 64, 256, 512
n_heads = 4        # Try: 2, 8, 16
n_layers = 2       # Try: 1, 3, 6
max_seq_len = 256  # Try: 128, 512
```

Larger values = slower but more powerful.

### "I want to train on real data"
See `train_tokenizer.py`:
- Downloads ~5MB of German public domain text from Project Gutenberg
- Trains BPE tokenizer
- Measures efficiency (fertility) on compound words
- Benchmarks different vocab sizes

---

## Key Concepts (30 second versions)

### Byte-Pair Encoding (BPE)
Start with characters. Repeatedly merge the most frequent adjacent pair. Result: efficient tokens.

Example: "hello" → ['h','e','l','l','o'] → ['h','el','l','o'] → ['h','ell','o'] → ... → [1 token]

### Multi-Head Attention
Each token computes similarity to every other token (scaled dot-product). Multiple heads do this in parallel with different projection weights.

Math: Attention(Q,K,V) = softmax(QK^T/√d_k)V

### Positional Encoding
Transformers don't know token order. Add sine/cosine signals at different frequencies so model can learn positions.

### Residual Connections
Add the input to the output of each layer. This helps:
- Train deeper networks
- Prevent vanishing gradients
- Maintain identity mapping

---

## Architecture at a Glance

```
German text
    ↓
[BPE Tokenizer] → token IDs
    ↓
[Token Embedding + Positional Encoding] → embeddings with position
    ↓
[Transformer Block 1]
  ├─ Multi-Head Attention (4 heads)
  ├─ + Residual + LayerNorm
  ├─ Feed-Forward (expand & contract)
  └─ + Residual + LayerNorm
    ↓
[Transformer Block 2]
  (same as Block 1)
    ↓
[Output Projection] → vocabulary logits
    ↓
Next token prediction / Text generation
```

---

## Performance

On standard CPU:

| Operation | Time |
|-----------|------|
| Train tokenizer | 0.5s |
| Forward pass | 10ms |
| Generate 10 tokens | 100ms |

(GPU would be 10-50× faster)

---

## References

- **Attention Is All You Need** (Vaswani et al. 2017)
  - Original Transformer paper
  - Introduced multi-head attention

- **Neural Machine Translation by Attention** (Bahdanau et al. 2014)
  - First attention mechanism paper
  - Simpler than multi-head, good starting point

- **BPE: Neural Machine Translation of Rare Words** (Sennrich et al. 2015)
  - Byte-Pair Encoding explanation
  - Why it works better than character-level

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch
```

### "AttributeError: 'BPETokeniser' has no attribute 'vocab'"
Use `token_to_id` instead:
```python
vocab_size = len(tokenizer.token_to_id)
```

### "CUDA out of memory"
Reduce model size in `main.py`:
```python
d_model = 64  # Instead of 128
n_heads = 2   # Instead of 4
```

---

## Next Steps

1. ✅ Run `python main.py` (see it work)
2. ✅ Read `PIPELINE.md` (understand architecture)
3. ✅ Run `python demo.py` (understand tokenization)
4. ✅ Run `python mha_examples.py` (understand attention)
5. 📝 Read the source code (each file is well-commented)
6. 🔧 Modify hyperparameters and see what happens
7. 📚 Extend it: better sampling, larger model, more layers

---

**Questions?** Check the comments in each `.py` file—they're detailed!

**Want to go deep?** Read PIPELINE.md for the complete architecture explanation.
