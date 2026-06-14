# Multi-Head Attention from Scratch: Complete Guide

## The Core Concept: Scaled Dot-Product Attention

Attention is a mechanism that answers: **"Which parts of the input should I focus on?"**

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

### What are Q, K, V?

| Component | Meaning | Analogy |
|-----------|---------|---------|
| **Query (Q)** | What am I looking for? | A search query |
| **Key (K)** | What can I offer? | Document titles/tags |
| **Value (V)** | What information do I have? | Document content |

**Example:** In a sentence "The cat sat on the mat"
- When processing "sat", the Query asks: "What does this verb relate to?"
- Keys from all words answer: "Here's what I represent" (cat, mat, on, etc.)
- Values provide: "Here's my actual representation/meaning"

### The Math Step-by-Step

1. **Compute similarity:** `Q * K^T` produces a score for each query-key pair
   - High score = "these are related"
   - Low score = "these are unrelated"

2. **Scale by sqrt(d_k):** Prevents scores from becoming too extreme
   - Keeps gradients stable during backprop
   - d_k = embedding_dim / num_heads

3. **Softmax:** Converts scores to probabilities (all sum to 1)
   - Strong focus on few high-scoring words
   - Weak focus on low-scoring words

4. **Weight values:** Multiply each value by its attention weight
   - Results in a weighted combination of all values
   - High attention = this value contributes more


## Multi-Head Attention: Why Multiple Heads?

Instead of one big attention head, we split into `n_heads` parallel heads.

**Benefits:**
- 🧠 **Diversity:** Each head learns different semantic relationships
  - Head 1: syntax ("singular vs plural")
  - Head 2: semantics ("related concepts")
  - Head 3: grammar ("subject-verb agreement")
- ⚡ **Efficiency:** Same computation cost (d_k = d_model / n_heads)
- 🎯 **Expressive power:** Concatenating heads > single large head


### Visual: 8-Head Attention (d_model=512)

```
Input: [batch=2, seq_len=10, d_model=512]
                    ↓
        Split into 8 heads (d_k=64 each)
                    ↓
        [batch=2, n_heads=8, seq_len=10, d_k=64]
                    ↓
    Compute scaled dot-product in each head (in parallel!)
                    ↓
        Concatenate: [batch=2, seq_len=10, 512]
                    ↓
    Final projection: [batch=2, seq_len=10, 512]
```


## Positional Encoding: Giving Position Information

Transformers have **no recurrence**—they process all tokens in parallel.
Problem: "cat sat dog" and "dog sat cat" would produce identical results!
Solution: **Positional encoding** adds position information to embeddings.

### Sine-Cosine Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))     # Even dimensions
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))    # Odd dimensions
```

**Why sine and cosine?**
- 🌊 Different frequencies for different dimension pairs
  - Early dimensions: change every position (high frequency)
  - Later dimensions: change slowly (low frequency)
- 📐 Relative positions are learnable
  - PE(pos+k) can be computed from PE(pos) via linear combination
- ∞ Works for any sequence length (not limited to max_seq_len)

### Visual: PE for seq_len=5

```
Position 0: [0.0,  1.0,  0.0,  1.0,  ...]  (baseline)
Position 1: [0.84, 0.54, 0.07, 1.0,  ...]  (shifted)
Position 2: [0.91, -0.41, 0.14, 0.99, ...] (shifted more)
...

Each position gets a unique "fingerprint"
that gets added to token embeddings.
```


## Code Walkthrough

### 1. Positional Encoding

```python
def forward(self, x):
    # x: [batch_size, seq_len, d_model]
    return x + self.pe[:x.size(1)]  # Add positional info
```

Simple! Just add precomputed positional encodings to embeddings.

### 2. Scaled Dot-Product Attention

```python
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
# [batch, n_heads, seq_len, seq_len] - attention matrix

attention = torch.softmax(scores, dim=-1)
# Normalize: each row sums to 1

output = torch.matmul(attention, value)
# Weight values: [batch, n_heads, seq_len, d_v]
```

### 3. Multi-Head Attention

```python
# Project to Q, K, V
Q = self.W_q(query)  # [batch, seq_len, d_model]
# Reshape to split heads
Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
# [batch, n_heads, seq_len, d_k]

# Apply attention
attn_output, _ = self.attention(Q, K, V)
# [batch, n_heads, seq_len, d_k]

# Concatenate heads
output = attn_output.transpose(1, 2).contiguous()
output = output.view(batch_size, -1, self.d_model)
# [batch, seq_len, d_model]

# Final projection
output = self.W_o(output)
```

## Parameters Explained

| Parameter | Purpose | Typical Values |
|-----------|---------|-----------------|
| `d_model` | Embedding dimension | 512, 768, 1024 |
| `n_heads` | Number of attention heads | 8, 12, 16 |
| `d_k` | Dim per head (d_model/n_heads) | 64, 96 |
| `max_seq_len` | Max sequence length for PE | 512, 2048 |


## Usage Example

```python
from multi_head_attention import MultiHeadAttention, PositionalEncoding

# Initialize
pe = PositionalEncoding(d_model=512, max_seq_len=512)
mha = MultiHeadAttention(d_model=512, n_heads=8)

# Dummy input
x = torch.randn(batch_size=2, seq_len=10, d_model=512)

# Add positions
x = pe(x)

# Self-attention (Q=K=V)
output = mha(x, x, x)

# Cross-attention (different K, V)
context = torch.randn(2, 10, 512)
output = mha(x, context, context)
```

## Key Takeaways

✅ **Attention solves "what to focus on"** by comparing queries to keys
✅ **Multi-head enables diverse pattern learning** in parallel
✅ **Positional encoding injects position info** into embeddings
✅ **Scaled dot-product** prevents numerical instability
✅ **Linear projections** (W_q, W_k, W_v) make Q, K, V learnable

## Next Steps

- Implement Feed-Forward networks (FFN)
- Build a complete Transformer encoder block (MHA + FFN + residual + norm)
- Stack blocks to create a full Transformer
- Train on next-token prediction
