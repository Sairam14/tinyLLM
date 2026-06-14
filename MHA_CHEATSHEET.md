# Multi-Head Attention: Quick Reference

## Core Formula

```
Attention(Q, K, V) = softmax(Q * K^T / √d_k) * V
```

## Dimensions

```
Input:    [batch_size, seq_len, d_model]
    ↓ (project to Q, K, V)
    [batch_size, seq_len, d_model]
    ↓ (reshape to split heads)
    [batch_size, n_heads, seq_len, d_k]  where d_k = d_model / n_heads
```

## Step-by-Step

### 1. Project Input
```python
Q = W_q @ x                    # Linear projection
K = W_k @ x
V = W_v @ x
```

### 2. Split into Heads
```python
# Reshape: [batch, seq_len, d_model] → [batch, seq_len, n_heads, d_k]
Q = Q.view(batch, seq_len, n_heads, d_k)
#  Then transpose: [batch, n_heads, seq_len, d_k]
Q = Q.transpose(1, 2)
```

### 3. Compute Attention
```python
scores = Q @ K^T / √d_k           # [batch, n_heads, seq_len, seq_len]
attention = softmax(scores)        # Normalize
output = attention @ V             # Weight values
```

### 4. Concatenate Heads
```python
# [batch, n_heads, seq_len, d_k] → [batch, seq_len, d_model]
output = output.transpose(1, 2)                    # Reorder
output = output.view(batch, seq_len, d_model)     # Reshape
output = W_o @ output                             # Output projection
```

## Q, K, V Intuition

| Role | Represents | Used in |
|------|-----------|---------|
| **Query** | "What am I looking for?" | decoder (decoder-encoder attn) |
| **Key** | "What can I offer?" | encoder (all-to-all visibility) |
| **Value** | "Here's my content" | encoder (all-to-all visibility) |

### Common Patterns

| Attention Type | Q | K | V | Use Case |
|---|---|---|---|---|
| **Self** | x | x | x | Encoder: see all context |
| **Causal** | x | x | x | Decoder: can't see future |
| **Cross** | decoder | encoder | encoder | Seq2seq: decoder queries encoder |

## Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Applied before attention:
```python
x_embedded = embedding(x)
x_with_pos = x_embedded + positional_encoding(x)
```

## Scaling Factor √d_k

**Why scale by √d_k?**

- Attention scores ~ Q @ K^T have variance ~ d_k
- Unnormalized scores become too extreme (softmax ≈ one-hot)
- Scaling keeps gradients stable

**Example:**
```python
d_k = 512 / 8 = 64              # For 8 heads
scale = 1 / √64 = 1 / 8 = 0.125
scores = scores * 0.125         # Prevents saturation
```

## Common Hyperparameters

| Parameter | Typical | Range | Effect |
|-----------|---------|-------|--------|
| `d_model` | 512, 768 | 128–2048 | Total embedding dimension |
| `n_heads` | 8, 12 | 4–32 | Must divide d_model evenly |
| `d_k` | d_model/n_heads | — | Dimension per head |
| `d_v` | d_model/n_heads | — | Usually same as d_k |
| `max_seq_len` | 512, 2048 | — | Max positional encoding length |

**Relationship:** d_k = d_model / n_heads

Example: d_model=512, n_heads=8 → d_k=64

## Causal Mask

Prevents attending to future tokens (for decoder training):

```python
# Create mask: lower triangular = 1, upper = 0
mask = torch.tril(torch.ones(seq_len, seq_len))

# Apply: set future positions to -inf before softmax
scores[mask == 0] = float('-inf')
```

Result:
```
Position 0 can attend to: [0]
Position 1 can attend to: [0, 1]
Position 2 can attend to: [0, 1, 2]
...
Position n can attend to: [0, 1, ..., n]
```

## Complexity Analysis

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Attention scores | O(n²d_k) | n=seq_len, quadratic in sequence |
| Softmax | O(n²) | Linear in scores |
| Value weighting | O(n²d_v) | Quadratic bottleneck |
| **Total** | **O(n²d_model)** | Scales poorly with long sequences |

## Implementation Checklist

- [ ] Create W_q, W_k, W_v, W_o linear layers
- [ ] Project input through W_q, W_k, W_v
- [ ] Reshape to split heads: [batch, seq_len, d_model] → [batch, n_heads, seq_len, d_k]
- [ ] Compute scores: Q @ K^T
- [ ] Scale by 1/√d_k
- [ ] Apply optional mask (causal, padding)
- [ ] Apply softmax
- [ ] Multiply by V
- [ ] Reshape back: [batch, n_heads, seq_len, d_k] → [batch, seq_len, d_model]
- [ ] Project through W_o

## Common Pitfalls

❌ **Forgetting the scale factor** → Attention becomes one-hot, gradients vanish

❌ **Wrong reshape order** → Mixing up batch and head dimensions

❌ **Applying softmax on wrong dimension** → Attention doesn't sum to 1 across keys

❌ **Missing positional encoding** → Model treats "cat dog" same as "dog cat"

❌ **No mask in decoder** → Cheating during training (looking at future)

## Torch Operations Cheatsheet

```python
# Transpose last two dimensions
K_T = K.transpose(-2, -1)

# Matrix multiply
scores = torch.matmul(Q, K_T)

# Apply softmax over last dimension
attn = torch.softmax(scores, dim=-1)

# Reshape with batch dimension preserved
x = x.view(batch, seq_len, n_heads, d_k)

# Transpose middle dimensions
x = x.transpose(1, 2)

# Flatten last two dimensions
x = x.contiguous().view(batch, seq_len, d_model)

# Masked fill (set values where condition is false)
x[mask == 0] = float('-inf')
# or
x = x.masked_fill(mask == 0, float('-inf'))
```

## Approximating Attention

For long sequences (O(n²) is expensive), approximations exist:

- **Linear Attention**: Use kernels instead of softmax
- **Sparse Attention**: Attend to subset of positions
- **Local Attention**: Attend to nearby tokens only
- **Performer**: Fast kernel-based approximation
- **Flash Attention**: Optimized GPU implementation

Research area: making attention faster without losing expressiveness.
