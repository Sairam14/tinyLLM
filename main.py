#!/usr/bin/env python3
"""
tinyLLM — Unified Entry Point
Connects all pieces: Tokenization (Part 1) + Multi-Head Attention (Part 2)

This is the main pipeline showing how to:
1. Tokenize German text (BPE)
2. Embed tokens with positional encoding
3. Process through multi-head attention
4. Generate predictions with a tiny transformer

Run this to see the complete pipeline in action.
"""

import torch
import torch.nn as nn
from bpe_tokenizer import BPETokeniser
from multi_head_attention import MultiHeadAttention, PositionalEncoding
from mha_examples import SimpleLanguageModel, create_causal_mask


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: German Text Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def setup_tokenizer():
    """Train a BPE tokenizer on German text."""
    print("\n" + "="*80)
    print("PART 1: TOKENIZATION — German BPE Tokenizer")
    print("="*80)

    # German text examples
    corpus = """
    Das ist ein Test für den BPE-Tokenizer.
    Deutsch ist eine Sprache mit Umlauten wie ä, ö, ü.
    Die Bildung ist wichtig für die Zukunft.
    Guten Morgen, wie geht es dir heute?
    Das Donaudampfschifffahrtsgesellschaftskapitän ist ein langes deutsches Wort.
    Ich liebe die deutsche Sprache und Kultur.
    München, Berlin und Hamburg sind große Städte.
    Die Schönheit der Natur ist beeindruckend.
    """

    # Create and train tokenizer
    vocab_size = 256
    tokenizer = BPETokeniser(vocab_size=vocab_size)
    tokenizer.train(corpus, verbose=False)

    print(f"✓ Tokenizer trained with vocab size: {len(tokenizer.token_to_id)}")
    return tokenizer


def test_tokenization(tokenizer):
    """Demonstrate tokenization on German text."""
    print("\n" + "-"*80)
    print("Tokenization Examples")
    print("-"*80)

    examples = [
        "Das ist ein Test.",
        "Guten Morgen!",
        "Donaudampfschifffahrtsgesellschaftskapitän",
    ]

    for text in examples:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        fertility = len(tokens) / len(text.split())

        print(f"\nText:     {text}")
        print(f"Tokens:   {tokens}")
        print(f"Decoded:  {decoded}")
        print(f"Fertility: {fertility:.2f} tokens/word")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Multi-Head Attention Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class GermanLanguageModel(nn.Module):
    """
    Complete tiny LLM for German text.

    Architecture:
    - Token embedding
    - Positional encoding
    - 2-layer transformer (multi-head attention + feed-forward)
    - Output projection to vocabulary
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, n_heads) for _ in range(n_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model),
            ) for _ in range(n_layers)
        ])

        self.layer_norms_attn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.layer_norms_ffn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, is_training=True):
        """
        Forward pass through the model.

        Args:
            token_ids: [batch_size, seq_len] - token indices
            is_training: if False, apply causal mask (autoregressive)

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed tokens and add positional encoding
        x = self.token_embedding(token_ids)  # [batch, seq_len, d_model]
        x = self.positional_encoding(x)

        # Create causal mask (prevents attending to future tokens)
        device = x.device
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len, device)

        # Apply transformer layers
        for attn_layer, ffn_layer, norm_attn, norm_ffn in zip(
            self.attention_layers, self.ffn_layers, self.layer_norms_attn, self.layer_norms_ffn
        ):
            # Multi-head attention with residual connection
            attn_out = attn_layer(x, x, x, mask=causal_mask)
            x = x + attn_out
            x = norm_attn(x)

            # Feed-forward with residual connection
            ffn_out = ffn_layer(x)
            x = x + ffn_out
            x = norm_ffn(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        return logits


def setup_model(tokenizer):
    """Initialize the language model."""
    print("\n" + "="*80)
    print("PART 2: MULTI-HEAD ATTENTION — Language Model Setup")
    print("="*80)

    vocab_size = len(tokenizer.token_to_id)
    model = GermanLanguageModel(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=256
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Embedding dim: 128")
    print(f"  - Attention heads: 4")
    print(f"  - Layers: 2")
    print(f"  - Total parameters: {total_params:,}")

    return model


def test_forward_pass(model, tokenizer):
    """Test model on German text."""
    print("\n" + "-"*80)
    print("Forward Pass Example")
    print("-"*80)

    # Encode German text
    text = "Das ist ein Test."
    token_ids = tokenizer.encode(text)

    print(f"\nInput text: {text}")
    print(f"Token IDs: {token_ids}")

    # Convert to tensor (add batch dimension)
    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    print(f"Tensor shape: {input_tensor.shape}")

    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor, is_training=False)

    print(f"Output logits shape: {logits.shape}")
    print(f"✓ Model processed successfully")

    # Show predicted next token at each position
    print(f"\nNext token predictions:")
    for i, token_id in enumerate(token_ids):
        next_token_logits = logits[0, i, :]
        predicted_token_id = next_token_logits.argmax().item()
        predicted_token = tokenizer.decode([predicted_token_id])
        actual_next = tokenizer.decode([token_ids[i+1]]) if i+1 < len(token_ids) else "⏹️  (end)"

        current_token = tokenizer.decode([token_id])
        print(f"  After '{current_token:15}' → predict '{predicted_token:15}' (actual: {actual_next})")


def generate_text(model, tokenizer, prompt, max_length=10):
    """Generate text using the model."""
    print("\n" + "-"*80)
    print("Text Generation Example")
    print("-"*80)

    # Start with prompt
    token_ids = tokenizer.encode(prompt)
    print(f"Prompt: {prompt}")
    print(f"Initial tokens: {token_ids}")

    generated_tokens = list(token_ids)

    # Generate new tokens
    print(f"\nGenerating {max_length} new tokens...")
    with torch.no_grad():
        for step in range(max_length):
            # Model expects batch dimension
            input_tensor = torch.tensor([generated_tokens], dtype=torch.long)

            # Get predictions for last position
            logits = model(input_tensor, is_training=False)
            last_logits = logits[0, -1, :]  # Last position

            # Sample next token (greedy)
            next_token_id = last_logits.argmax().item()
            generated_tokens.append(next_token_id)

            next_token_str = tokenizer.decode([next_token_id])
            print(f"  Step {step+1}: generated token {next_token_id} ({next_token_str})")

    # Decode full sequence
    full_text = tokenizer.decode(generated_tokens)
    print(f"\nFull generated text: {full_text}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run the complete pipeline."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  tinyLLM: Building a Tiny Transformer from Scratch".center(78) + "█")
    print("█" + "  German Tokenization + Multi-Head Attention Pipeline".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    # Step 1: Setup tokenizer
    tokenizer = setup_tokenizer()
    test_tokenization(tokenizer)

    # Step 2: Setup model
    model = setup_model(tokenizer)

    # Step 3: Test forward pass
    test_forward_pass(model, tokenizer)

    # Step 4: Generate text
    generate_text(model, tokenizer, prompt="Das ist", max_length=5)

    print("\n" + "="*80)
    print("✓ Complete pipeline executed successfully!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review multi_head_attention.py for core MHA implementation")
    print("  2. Review mha_examples.py for advanced patterns and examples")
    print("  3. Check bpe_tokenizer.py for tokenization details")
    print("  4. Run: python mha_examples.py  (for 6 detailed attention examples)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
