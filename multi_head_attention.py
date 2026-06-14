"""
Multi-Head Attention in 80 Lines of PyTorch
Implements MHA from scratch with positional encoding
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings.
    Uses sine/cosine patterns so the model can learn relative positions.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
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

    Q (Query): What am I looking for?
    K (Key): What can I offer?
    V (Value): What information do I have?

    The sqrt(d_k) scaling prevents attention weights from becoming too sharp/small.
    """
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, n_heads, seq_len, d_k]
            key:   [batch_size, n_heads, seq_len, d_k]
            value: [batch_size, n_heads, seq_len, d_v]
            mask:  Optional attention mask
        """
        d_k = query.size(-1)

        # 1. Compute attention scores: Q * K^T
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. Apply mask (for causal attention in decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 3. Apply softmax to get attention weights
        attention = torch.softmax(scores, dim=-1)

        # 4. Weight values by attention
        output = torch.matmul(attention, value)

        return output, attention


class MultiHeadAttention(nn.Module):
    """
    MHA = Concat(head_1, ..., head_h) * W^O

    Each head independently learns different representation subspaces.
    Unlike one huge attention head, multiple heads allow parallel learning.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: Optional causal mask for decoder
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)

        # 1. Linear projections and reshape for multi-head
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k]
        # -> [batch_size, n_heads, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Apply scaled dot-product attention
        attn_output, _ = self.attention(Q, K, V, mask)

        # 3. Concatenate heads: [batch_size, n_heads, seq_len, d_k]
        # -> [batch_size, seq_len, n_heads, d_k] -> [batch_size, seq_len, d_model]
        output = attn_output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        # 4. Final linear projection
        output = self.W_o(output)

        return output


# Demo usage
if __name__ == "__main__":
    # Hyperparameters
    d_model = 512      # Embedding dimension
    n_heads = 8        # Number of attention heads
    seq_len = 10       # Sequence length
    batch_size = 2     # Batch size

    # Create components
    positional_encoding = PositionalEncoding(d_model, max_seq_len=512)
    mha = MultiHeadAttention(d_model, n_heads)

    # Dummy input: [batch_size, seq_len, d_model]
    x = torch.randn(batch_size, seq_len, d_model)

    # Add positional encoding
    x_with_pos = positional_encoding(x)

    # Apply multi-head attention (self-attention: Q=K=V)
    output = mha(x_with_pos, x_with_pos, x_with_pos)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"PE contribution: {(positional_encoding.pe[:seq_len] ** 2).mean():.4f}")
