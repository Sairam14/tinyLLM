"""
Test Multi-Head Attention with German Text
Combines BPE tokenizer (Part 1) + MHA (Part 2)
"""

import torch
import torch.nn as nn
from bpe_tokenizer import BPETokeniser
from multi_head_attention import MultiHeadAttention, PositionalEncoding
import json


class MHATokenizerPipeline(nn.Module):
    """
    End-to-end pipeline: German text → tokens → embeddings → MHA
    """
    def __init__(self, tokenizer, d_model, n_heads, max_seq_len=512):
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.token_to_id)
        self.d_model = d_model

        # Embedding and positional encoding
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Multi-head attention
        self.mha = MultiHeadAttention(d_model, n_heads)

        # Optional: project back to vocabulary for analysis
        self.output_projection = nn.Linear(d_model, self.vocab_size)

    def forward(self, text):
        """
        Args:
            text: str (German text)
        Returns:
            logits: [1, seq_len, vocab_size] - predictions for each token
            token_ids: token IDs used
            embeddings: after positional encoding
            attention_output: after MHA
        """
        # Tokenize
        token_ids = self.tokenizer.encode(text)
        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)  # [1, seq_len]

        # Embed
        embeddings = self.embedding(token_ids_tensor)  # [1, seq_len, d_model]

        # Add positional encoding
        with_pos = self.positional_encoding(embeddings)  # [1, seq_len, d_model]

        # Apply MHA (self-attention)
        attention_output = self.mha(with_pos, with_pos, with_pos)  # [1, seq_len, d_model]

        # Project to vocabulary
        logits = self.output_projection(attention_output)  # [1, seq_len, vocab_size]

        return logits, token_ids, embeddings, attention_output


def test_basic_mha():
    """Test MHA with a simple German sentence"""
    print("=" * 80)
    print("Test 1: Basic German Sentence")
    print("=" * 80)

    # Load tokenizer
    try:
        tokenizer = BPETokeniser.load('german_bpe_8k.json')
    except FileNotFoundError:
        print("⚠️  Trained tokenizer not found. Using demo mode.")
        print("   Run: python train_tokenizer.py")
        return

    # Initialize pipeline
    d_model = 256
    n_heads = 4
    pipeline = MHATokenizerPipeline(tokenizer, d_model, n_heads)

    # Test German text
    german_text = "Der Katze sitzt auf der Matte."
    print(f"\nInput text: {german_text}")

    # Get all outputs
    logits, token_ids, embeddings, attention_output = pipeline(german_text)

    # Decode tokens
    tokens = [tokenizer.id_to_token.get(tid, f"<unk-{tid}>") for tid in token_ids]
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Num tokens: {len(token_ids)}")

    # Show shapes
    print(f"\nTensor shapes:")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  After MHA shape: {attention_output.shape}")
    print(f"  Logits shape: {logits.shape}")

    # Show attention statistics
    print(f"\nAttention statistics:")
    print(f"  Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    print(f"  After MHA range: [{attention_output.min():.3f}, {attention_output.max():.3f}]")
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

    # Top predictions for last token
    last_token_logits = logits[0, -1, :]
    top_k = 5
    top_logits, top_indices = torch.topk(last_token_logits, top_k)
    top_probs = torch.softmax(top_logits, dim=0)

    print(f"\nTop {top_k} next-token predictions after '{tokens[-1]}':")
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        pred_token = tokenizer.id_to_token.get(idx.item(), f"<unk-{idx}>")
        print(f"  {i+1}. '{pred_token}' ({prob.item():.3%})")


def test_compound_words():
    """Test MHA with German compound words (why Part 1 matters!)"""
    print("\n" + "=" * 80)
    print("Test 2: German Compound Words (Tokenizer Advantage)")
    print("=" * 80)

    try:
        tokenizer = BPETokeniser.load('german_bpe_8k.json')
    except FileNotFoundError:
        print("⚠️  Trained tokenizer not found.")
        return

    d_model = 256
    n_heads = 4
    pipeline = MHATokenizerPipeline(tokenizer, d_model, n_heads)

    # Compound word examples
    compounds = [
        "Donaudampfschifffahrtsgesellschaft",  # Famous German compound
        "Freundschaftsbeziehung",
        "Gemäldeausstellung",
    ]

    print("\nCompound word tokenization:")
    for compound in compounds:
        token_ids = tokenizer.encode(compound)
        tokens = [tokenizer.id_to_token.get(tid, f"<unk-{tid}>") for tid in token_ids]

        print(f"\n  Word: {compound}")
        print(f"    Tokens: {tokens}")
        print(f"    Num tokens: {len(token_ids)}")
        print(f"    Fertility: {len(token_ids) / 1:.2f} tokens/word")

        # Get MHA output
        logits, _, _, attn_out = pipeline(compound)
        print(f"    Attention output shape: {attn_out.shape}")


def test_longer_text():
    """Test MHA with longer German text"""
    print("\n" + "=" * 80)
    print("Test 3: Longer German Text")
    print("=" * 80)

    try:
        tokenizer = BPETokeniser.load('german_bpe_8k.json')
    except FileNotFoundError:
        print("⚠️  Trained tokenizer not found.")
        return

    d_model = 256
    n_heads = 4
    pipeline = MHATokenizerPipeline(tokenizer, d_model, n_heads)

    # Goethe quote
    german_text = """Zwei Dinge erfüllen das Gemüt mit immer neuer und zunehmender Bewunderung
                     und Ehrfurcht, je öfter und anhaltender sich das Nachdenken damit beschäftigt:
                     der bestirnte Himmel über mir und das moralische Gesetz in mir."""

    # Tokenize
    token_ids = tokenizer.encode(german_text)
    tokens = [tokenizer.id_to_token.get(tid, f"<unk-{tid}>") for tid in token_ids]

    print(f"\nInput: {german_text[:100]}...")
    print(f"Total tokens: {len(token_ids)}")
    print(f"Fertility: {len(token_ids) / len(german_text.split()):.2f} tokens/word")

    # Get MHA output
    logits, _, embeddings, attention_output = pipeline(german_text)

    print(f"\nProcessing through MHA:")
    print(f"  Input embeddings: {embeddings.shape}")
    print(f"  After MHA: {attention_output.shape}")
    print(f"  Output logits: {logits.shape}")

    # Analyze attention
    print(f"\nAttention statistics:")
    print(f"  Embedding variance: {embeddings.var():.4f}")
    print(f"  After MHA variance: {attention_output.var():.4f}")

    # Decode and show a sample
    print(f"\nSample tokens (first 10):")
    for i, token in enumerate(tokens[:10]):
        print(f"  {i}: {token}")


def test_attention_flow():
    """Detailed trace of data flow through MHA"""
    print("\n" + "=" * 80)
    print("Test 4: Detailed Attention Flow Analysis")
    print("=" * 80)

    try:
        tokenizer = BPETokeniser.load('german_bpe_8k.json')
    except FileNotFoundError:
        print("⚠️  Trained tokenizer not found.")
        return

    d_model = 128  # Smaller for easier visualization
    n_heads = 4
    pipeline = MHATokenizerPipeline(tokenizer, d_model, n_heads)

    german_text = "Das ist ein Test ."
    print(f"\nText: {german_text}")

    # Step-by-step
    token_ids = tokenizer.encode(german_text)
    tokens = [tokenizer.id_to_token.get(tid, f"<unk-{tid}>") for tid in token_ids]

    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}\n")

    # Tokenize
    token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)
    seq_len = token_ids_tensor.shape[1]

    print("Data Flow:")
    print(f"1. Token IDs: {token_ids_tensor.shape}")

    # Embed
    embeddings = pipeline.embedding(token_ids_tensor)
    print(f"2. Embeddings: {embeddings.shape}")
    print(f"   Value range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    print(f"   Norm per token (first 3):")
    for i in range(min(3, seq_len)):
        norm = torch.norm(embeddings[0, i])
        print(f"     Token {i} ({tokens[i]}): norm={norm:.3f}")

    # Add positional encoding
    with_pos = pipeline.positional_encoding(embeddings)
    print(f"3. + Positional Encoding: {with_pos.shape}")
    print(f"   Value range: [{with_pos.min():.3f}, {with_pos.max():.3f}]")
    pos_contribution = (with_pos - embeddings).abs().mean()
    print(f"   PE contribution (avg): {pos_contribution:.4f}")

    # MHA
    attention_output = pipeline.mha(with_pos, with_pos, with_pos)
    print(f"4. After MHA: {attention_output.shape}")
    print(f"   Value range: [{attention_output.min():.3f}, {attention_output.max():.3f}]")

    # Output projection
    logits = pipeline.output_projection(attention_output)
    print(f"5. Output logits: {logits.shape}")
    print(f"   Value range: [{logits.min():.3f}, {logits.max():.3f}]")

    # Softmax to probabilities
    probs = torch.softmax(logits[0, -1, :], dim=0)
    print(f"6. Final probabilities (last position): shape={probs.shape}")
    print(f"   Max probability: {probs.max():.3%}")
    print(f"   Top-5 predictions:")
    top_k = 5
    top_probs, top_indices = torch.topk(probs, top_k)
    for prob, idx in zip(top_probs, top_indices):
        pred_token = tokenizer.id_to_token.get(idx.item(), f"<unk-{idx}>")
        print(f"     {pred_token}: {prob.item():.3%}")


def test_compare_with_without_mha():
    """Compare embeddings before/after MHA"""
    print("\n" + "=" * 80)
    print("Test 5: Embeddings Before vs After MHA")
    print("=" * 80)

    try:
        tokenizer = BPETokeniser.load('german_bpe_8k.json')
    except FileNotFoundError:
        print("⚠️  Trained tokenizer not found.")
        return

    d_model = 256
    n_heads = 4
    pipeline = MHATokenizerPipeline(tokenizer, d_model, n_heads)

    german_text = "Der Hund spielt mit dem Ball im Garten ."
    token_ids = tokenizer.encode(german_text)
    tokens = [tokenizer.id_to_token.get(tid, f"<unk-{tid}>") for tid in token_ids]

    # Get embeddings
    token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)
    embeddings = pipeline.embedding(token_ids_tensor)
    embeddings_with_pos = pipeline.positional_encoding(embeddings)
    attention_output = pipeline.mha(embeddings_with_pos, embeddings_with_pos, embeddings_with_pos)

    print(f"Text: {german_text}")
    print(f"Tokens: {tokens}\n")

    print("Token-wise analysis (cosine similarity to first token):")
    first_embed = attention_output[0, 0, :]

    for i, token in enumerate(tokens):
        token_embed = attention_output[0, i, :]
        cos_sim = torch.cosine_similarity(first_embed.unsqueeze(0), token_embed.unsqueeze(0))
        print(f"  {i:2d}. '{token:20s}' → cosine_sim to token 0: {cos_sim.item():7.3f}")


if __name__ == "__main__":
    print("\n" + "🧪" * 40)
    print("Testing Multi-Head Attention with German Text")
    print("🧪" * 40 + "\n")

    test_basic_mha()
    test_compound_words()
    test_longer_text()
    test_attention_flow()
    test_compare_with_without_mha()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
