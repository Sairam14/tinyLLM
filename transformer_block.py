#!/usr/bin/env python3
"""
Part 3: Transformer Block — Complete LLM Building Block

Assembles the complete transformer block:
  1. Layer Normalisation (Pre-LayerNorm design)
  2. Multi-Head Attention (from Part 2)
  3. Residual Connections
  4. Feed-Forward Network (FFN)

The transformer block is the fundamental repeating unit in modern LLMs.
Stack N of these blocks and you have the core of GPT, LLaMA, or Gemma.

Architecture (Pre-LayerNorm):
  Input x
    ↓
  LayerNorm → MultiHeadAttention → Dropout → + Residual
    ↓
  LayerNorm → FFN (Linear→ReLU→Linear) → Dropout → + Residual
    ↓
  Output

This ensures stable gradient flow for training deep networks (Xiong et al., 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention import (
    MultiHeadAttention,
    PositionalEncoding,
)


class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network (FFN).

    Every token position gets processed through the same MLP:
      Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model)

    Expanded dimension (d_ff = 4 * d_model) adds nonlinearity and capacity.
    Applied independently to each position (hence "position-wise").
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        x = self.linear1(x)  # [batch, seq_len, d_ff]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)  # [batch, seq_len, d_model]
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with Pre-LayerNorm and residual connections.

    Pre-LayerNorm (used in modern LLMs):
      - Normalize BEFORE applying the sublayer (attention or FFN)
      - More stable gradient flow, easier to scale to depth
      - Better than Post-LayerNorm for deep networks (100+ layers)

    Key insight: Residual connections allow information to "bypass" layers,
    letting gradients flow directly back during backprop. Without residuals,
    deep networks become hard to train (vanishing gradients).
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Pre-LayerNorm: normalize before each sublayer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads)

        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        # Dropout for regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass with Pre-LayerNorm and residual connections.

        Args:
            x: [batch, seq_len, d_model] - input embeddings
            mask: [seq_len, seq_len] - attention mask (causal or padding)

        Returns:
            [batch, seq_len, d_model] - output
        """
        batch_size, seq_len, d_model = x.shape

        # ═══════════════════════════════════════════════════════════
        # Attention Sublayer with Pre-LN and Residual
        # ═══════════════════════════════════════════════════════════
        x_norm = self.norm1(x)  # [batch, seq_len, d_model]
        attn_out = self.attention(x_norm, x_norm, x_norm, mask=mask)
        attn_out = self.dropout1(attn_out)
        x = x + attn_out  # Residual: skip-connection

        # ═══════════════════════════════════════════════════════════
        # Feed-Forward Sublayer with Pre-LN and Residual
        # ═══════════════════════════════════════════════════════════
        x_norm = self.norm2(x)  # [batch, seq_len, d_model]
        ffn_out = self.ffn(x_norm)
        ffn_out = self.dropout2(ffn_out)
        x = x + ffn_out  # Residual: skip-connection

        return x


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal (triangular) attention mask.

    Prevents attention to future positions (for autoregressive generation).
    Tokens at position t can only attend to positions ≤ t.

    Shape: [seq_len, seq_len] where mask[i, j] = 0 if j > i (can attend)
                                                    -inf if j > i (cannot attend)

    Visualization for seq_len=4:
      pos: 0 1 2 3
      0    ✓ ✗ ✗ ✗   (pos 0 can only attend to itself)
      1    ✓ ✓ ✗ ✗   (pos 1 can attend to 0,1)
      2    ✓ ✓ ✓ ✗   (pos 2 can attend to 0,1,2)
      3    ✓ ✓ ✓ ✓   (pos 3 can attend to all)
    """
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    return mask


class SimpleTransformer(nn.Module):
    """Stacking multiple transformer blocks to build a complete LLM.

    A complete language model is just:
      1. Token embeddings
      2. Positional embeddings
      3. N transformer blocks
      4. Linear projection to vocabulary
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_blocks = n_blocks

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Stack transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        # Final layer norm (helps with training stability)
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """Forward pass through complete transformer model.

        Args:
            token_ids: [batch, seq_len] - token indices
            verbose: if True, print shape info for debugging

        Returns:
            logits: [batch, seq_len, vocab_size] - raw scores for each token
        """
        batch_size, seq_len = token_ids.shape

        # ═══════════════════════════════════════════════════════════
        # Embedding Layer
        # ═══════════════════════════════════════════════════════════
        x = self.token_embedding(token_ids)  # [batch, seq_len, d_model]
        x = self.positional_encoding(x)  # Add positional info

        if verbose:
            print(f"After embedding: {x.shape}")

        # ═══════════════════════════════════════════════════════════
        # Transformer Blocks
        # ═══════════════════════════════════════════════════════════
        causal_mask = create_causal_mask(seq_len, x.device)

        for i, block in enumerate(self.blocks):
            x = block(x, mask=causal_mask)
            if verbose:
                print(f"After block {i+1}: {x.shape}")

        # ═══════════════════════════════════════════════════════════
        # Output Projection
        # ═══════════════════════════════════════════════════════════
        x = self.final_norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits

    def generate(
        self,
        start_token_id: int,
        max_length: int = 20,
        temperature: float = 1.0,
    ) -> list:
        """Autoregressive generation: one token at a time.

        At each step:
          1. Forward pass through the model
          2. Get logits for the last position
          3. Sample next token
          4. Append to sequence and repeat
        """
        generated = [start_token_id]

        with torch.no_grad():
            for step in range(max_length):
                # Prepare input: everything generated so far
                input_ids = torch.tensor([generated], dtype=torch.long)

                # Forward pass
                logits = self(input_ids, verbose=False)  # [1, seq_len, vocab_size]

                # Get logits for last position
                last_logits = logits[0, -1, :] / temperature

                # Sample next token (greedy: argmax)
                next_token = torch.argmax(last_logits).item()
                generated.append(next_token)

        return generated


if __name__ == "__main__":
    print("="*80)
    print("PART 3: TRANSFORMER BLOCK — Complete LLM Building Block")
    print("="*80)

    # ─────────────────────────────────────────────────────────────────────────
    # Demo 1: Single Transformer Block
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("Demo 1: Single Transformer Block")
    print("-"*80)

    d_model = 8
    n_heads = 2
    seq_len = 10
    batch_size = 2

    block = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=32,
        dropout=0.1,
    )

    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - d_model: {d_model} (embedding dimension)")

    # Forward pass
    with torch.no_grad():
        output = block(x)

    print(f"Output shape: {output.shape}")
    print(f"✓ Single block works (input and output shape match)")

    # ─────────────────────────────────────────────────────────────────────────
    # Demo 2: Complete Transformer Model
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("Demo 2: Complete Transformer Model (2 blocks)")
    print("-"*80)

    vocab_size = 256
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=8,
        n_heads=2,
        n_blocks=2,
        max_seq_len=256,
    )

    # Random token sequence
    token_ids = torch.randint(0, vocab_size, (2, 5), dtype=torch.long)
    print(f"\nInput tokens: {token_ids}")
    print(f"Shape: {token_ids.shape} (batch=2, seq_len=5)")

    with torch.no_grad():
        logits = model(token_ids, verbose=True)

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"  - batch: 2")
    print(f"  - seq_len: 5")
    print(f"  - vocab_size: {vocab_size}")

    # Next token prediction
    print(f"\nNext token predictions:")
    for i in range(5):
        next_logits = logits[0, i, :]
        next_token = torch.argmax(next_logits).item()
        try:
            prob = torch.softmax(next_logits, dim=0)[next_token].item()
            conf_str = f"(conf: {prob:.2%})" if not (prob != prob) else "(untrained model)"
        except:
            conf_str = "(untrained model)"
        print(f"  Position {i}: predict token {next_token} {conf_str}")

    # ─────────────────────────────────────────────────────────────────────────
    # Demo 3: Text Generation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("Demo 3: Autoregressive Text Generation")
    print("-"*80)

    print("\nGenerating 20 tokens starting from token 5...")
    generated_ids = model.generate(start_token_id=5, max_length=20)
    print(f"Generated sequence: {generated_ids}")
    print(f"Total length: {len(generated_ids)} tokens")

    # ─────────────────────────────────────────────────────────────────────────
    # Model Architecture Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("Model Architecture")
    print("-"*80)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"\nArchitecture:")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Embedding dim: {d_model}")
    print(f"  - Attention heads: 2")
    print(f"  - Transformer blocks: 2")
    print(f"  - FFN hidden: {d_model * 4}")
    print(f"\nKey design choices:")
    print(f"  ✓ Pre-LayerNorm (normalize before each sublayer)")
    print(f"  ✓ Residual connections (skip connections)")
    print(f"  ✓ Causal masking (can't attend to future)")
    print(f"  ✓ Position-wise FFN (applied to each position)")
    print(f"\nThis is the core architecture of GPT, LLaMA, Gemma, etc.")
