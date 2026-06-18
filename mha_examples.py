"""
Advanced Multi-Head Attention Examples: Self-Attention, Cross-Attention, Causal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention, PositionalEncoding


def create_causal_mask(seq_len, device):
    """
    Causal mask: position i can only attend to positions <= i
    (Prevents "cheating" by looking at future tokens during training)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


class TransformerBlock(nn.Module):
    """
    A single Transformer encoder block: MHA -> FFN with residual & layer norm
    """
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, d_model]
        """
        # Multi-Head Attention with residual connection
        attn_out = self.mha(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-Forward with residual connection
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x


class SimpleLanguageModel(nn.Module):
    """
    Minimal Transformer-based language model for next-token prediction
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, is_training=True):
        """
        token_ids: [batch_size, seq_len] - token indices
        is_training: if False, apply causal mask
        """
        # Embed and add positions
        x = self.token_embedding(token_ids)  # [batch, seq_len, d_model]
        x = self.positional_encoding(x)

        # Apply causal mask during inference/training
        device = x.device
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len, device)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        # Project to vocabulary
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits


# Example 1: Self-Attention
def example_self_attention():
    print("=" * 60)
    print("Example 1: Self-Attention (Q=K=V)")
    print("=" * 60)

    d_model, n_heads = 256, 4
    mha = MultiHeadAttention(d_model, n_heads)

    x = torch.randn(2, 8, d_model)  # [batch=2, seq_len=8, d_model=256]
    output = mha(x, x, x)  # Each token attends to all tokens (including itself)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Each position can attend to ALL other positions")
    print()


# Example 2: Cross-Attention (Encoder-Decoder)
def example_cross_attention():
    print("=" * 60)
    print("Example 2: Cross-Attention (Q from decoder, K/V from encoder)")
    print("=" * 60)

    d_model, n_heads = 256, 4
    mha = MultiHeadAttention(d_model, n_heads)

    query = torch.randn(2, 5, d_model)      # Decoder output
    key_value = torch.randn(2, 8, d_model)  # Encoder output

    output = mha(query, key_value, key_value)

    print(f"Query shape (decoder): {query.shape}")
    print(f"Key/Value shape (encoder): {key_value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Decoder can attend to encoder outputs")
    print()


# Example 3: Causal Attention (Decoder only)
def example_causal_attention():
    print("=" * 60)
    print("Example 3: Causal Attention (Can't look at future)")
    print("=" * 60)

    d_model, n_heads = 256, 4
    mha = MultiHeadAttention(d_model, n_heads)

    x = torch.randn(1, 5, d_model)
    causal_mask = create_causal_mask(5, x.device)

    output = mha(x, x, x, mask=causal_mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"✓ Position 0 can only attend to position 0")
    print(f"✓ Position 4 can attend to positions 0-4 (not 5+)")
    print()


# Example 4: Transformer Block (MHA + FFN)
def example_transformer_block():
    print("=" * 60)
    print("Example 4: Transformer Block (Self-Attention + FFN)")
    print("=" * 60)

    d_model, n_heads = 256, 4
    block = TransformerBlock(d_model, n_heads, d_ff=512)

    x = torch.randn(2, 8, d_model)
    output = block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Added residual connections & layer normalization")
    print(f"✓ Feed-forward network expands to 512, projects back to 256")
    print()


# Example 5: Simple Language Model
def example_language_model():
    print("=" * 60)
    print("Example 5: Minimal Language Model")
    print("=" * 60)

    vocab_size = 100
    d_model = 256
    n_heads = 4
    n_layers = 2
    max_seq_len = 128

    model = SimpleLanguageModel(vocab_size, d_model, n_heads, n_layers, max_seq_len)

    # Random token sequence
    token_ids = torch.randint(0, vocab_size, (2, 10))  # [batch=2, seq_len=10]

    logits = model(token_ids, is_training=True)

    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Can predict next token at each position")
    print()

    # Show sample generation
    print("Sample generation:")
    with torch.no_grad():
        context = torch.tensor([[42, 17, 5]])  # Start with 3 tokens
        generated = list(context.squeeze().tolist())

        for _ in range(5):  # Generate 5 more tokens
            logits = model(context, is_training=False)
            next_token = logits[:, -1, :].argmax(dim=-1)  # Greedy sampling
            generated.append(next_token.item())
            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)

        print(f"Generated sequence: {generated}")


# Example 6: Attention Head Analysis
def example_attention_visualization():
    print("=" * 60)
    print("Example 6: Analyzing Attention Head Behavior")
    print("=" * 60)

    d_model, n_heads = 256, 4
    mha = MultiHeadAttention(d_model, n_heads)

    x = torch.randn(1, 5, d_model)

    # Manually extract attention weights
    batch_size = 1
    Q = mha.W_q(x).view(batch_size, -1, n_heads, 64).transpose(1, 2)
    K = mha.W_k(x).view(batch_size, -1, n_heads, 64).transpose(1, 2)
    V = mha.W_v(x).view(batch_size, -1, n_heads, 64).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / 8.0
    attention_weights = torch.softmax(scores, dim=-1)

    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"[batch=1, n_heads=4, seq_len=5, seq_len=5]")
    print()

    for head_idx in range(n_heads):
        weights = attention_weights[0, head_idx]
        print(f"Head {head_idx} - What does token at position 2 attend to?")
        for pos, weight in enumerate(weights[2]):
            print(f"  Position {pos}: {weight.item():.3f}")
    print()


if __name__ == "__main__":
    example_self_attention()
    example_cross_attention()
    example_causal_attention()
    example_transformer_block()
    example_language_model()
    example_attention_visualization()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)