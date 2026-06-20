"""
Multi-Head Attention in 80 Lines of PyTorch
Part 2 of "Building a Tiny German LLM from Scratch"

Connects directly to Part 1's BPE tokeniser: the 213-token vocabulary
trained on Kafka, Goethe, and Schiller feeds this attention mechanism.
The 31-token fragmentation of "Donaudampfschifffahrtsgesellschaft" is
exactly what this MHA layer has to reconstruct, token by token.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings.
    Uses sine/cosine patterns so the model can learn relative positions.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Fast-oscillating dimensions (low i) encode fine-grained nearby position.
    Slow-oscillating dimensions (high i) encode coarse long-range position.
    This is the continuous analogue of binary position counting.
    """
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [batch_size, seq_len, d_model]"""
        return x + self.pe[:x.size(1)]  # Broadcast positional encoding


class ScaledDotProductAttention(nn.Module):
    """
    Core attention mechanism: Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V

    Q (Query): What is this token looking for?
    K (Key): What does each token offer?
    V (Value): What does each token actually contribute if selected?

    The sqrt(d_k) scaling prevents dot products from growing too large as
    d_k increases (variance of Q.K grows with d_k), which would push
    softmax into saturated regions with near-zero gradients.
    """
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, n_heads, seq_len, d_k]
            key:   [batch_size, n_heads, seq_len, d_k]
            value: [batch_size, n_heads, seq_len, d_v]
            mask:  Optional attention mask. 1 = attend, 0 = block.
                   Used for causal masking (decoder) and/or padding masking.
        """
        d_k = query.size(-1)

        # 1. Compute attention scores: Q . K^T, scaled by sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. Apply mask BEFORE softmax: masked positions get -inf,
        #    which becomes exactly 0 probability after softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 3. Softmax converts scores to a probability distribution over
        #    the key/value positions, for each query position
        attention = torch.softmax(scores, dim=-1)

        # 4. Weighted sum of values, using attention probabilities as weights
        output = torch.matmul(attention, value)

        return output, attention


class MultiHeadAttention(nn.Module):
    """
    MultiHead(X) = Concat(head_1, ..., head_h) . W_O

    Each head independently learns a different relationship type --
    syntax, coreference, positional proximity, semantic similarity --
    operating in its own d_k-dimensional subspace. W_O mixes information
    across heads after concatenation; without it, head outputs would
    remain in disjoint, non-interacting subspaces of the output vector.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head -- computed once, reused everywhere

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: Optional mask, shape broadcastable to
                  [batch_size, n_heads, seq_len_q, seq_len_k]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_output, _ = self.attention(Q, K, V, mask)

        output = attn_output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        output = self.W_o(output)

        return output


if __name__ == "__main__":
    vocab_size = 213   # from Part 1: 169 merges + base chars = 213 tokens
    d_model = 8        # embedding dimension -- small enough to print
    n_heads = 2        # d_k = 8 // 2 = 4 per head
    seq_len = 10       # sequence length
    batch_size = 2     # batch size

    positional_encoding = PositionalEncoding(d_model, max_seq_len=512)
    mha = MultiHeadAttention(d_model, n_heads)

    x = torch.randn(batch_size, seq_len, d_model)
    x_with_pos = positional_encoding(x)
    output = mha(x_with_pos, x_with_pos, x_with_pos)

    print(f"Vocabulary size (from Part 1 tokeniser): {vocab_size}")
    print(f"d_model: {d_model}, n_heads: {n_heads}, d_k per head: {d_model // n_heads}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"PE contribution: {(positional_encoding.pe[:seq_len] ** 2).mean():.4f}")