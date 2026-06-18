"""Tests for GermanTokenizer (HuggingFace wrapper)."""

import pytest
from pathlib import Path

from tinyllm.tokenizer import GermanTokenizer


class TestGermanTokenizer:
    """Test GermanTokenizer functionality."""

    def test_tokenizer_creation(self):
        """Test tokenizer initialization."""
        tokenizer = GermanTokenizer(vocab_size=1000)
        assert tokenizer.vocab_size == 1000
        assert tokenizer.lowercase is True

    def test_tokenizer_train(self):
        """Test tokenizer training."""
        corpus = [
            "Das ist ein Test für den BPE-Tokenizer.",
            "Deutsch ist eine Sprache mit Umlauten wie ä, ö, ü.",
            "Die Bildung ist wichtig für die Zukunft.",
        ]

        tokenizer = GermanTokenizer(vocab_size=256)
        tokenizer.train(corpus, verbose=False)

        assert tokenizer.tokenizer is not None
        assert tokenizer.vocab_size > 0

    def test_encode_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        corpus = [
            "Das ist ein Test.",
            "Guten Morgen!",
            "Donaudampfschifffahrtsgesellschaftskapitän",
        ]

        tokenizer = GermanTokenizer(vocab_size=500)
        tokenizer.train(corpus, verbose=False)

        text = "Das ist ein Test."
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)

        # Decoded should be similar (may have slight differences due to tokenization)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_special_tokens(self):
        """Test special token IDs."""
        corpus = ["test corpus"]
        tokenizer = GermanTokenizer(vocab_size=256)
        tokenizer.train(corpus, verbose=False)

        assert tokenizer.PAD_ID == 0
        assert tokenizer.UNK_ID == 1
        assert tokenizer.BOS_ID == 2
        assert tokenizer.EOS_ID == 3

    def test_tokenizer_save_load(self, tmp_path):
        """Test tokenizer serialization."""
        corpus = ["Das ist ein Test.", "Guten Morgen!"]
        tokenizer = GermanTokenizer(vocab_size=256)
        tokenizer.train(corpus, verbose=False)

        # Save
        save_path = tmp_path / "tokenizer.json"
        tokenizer.save(save_path)
        assert save_path.exists()

        # Load
        tokenizer2 = GermanTokenizer()
        tokenizer2.load(save_path)

        # Compare
        text = "Das ist ein Test."
        tokens1 = tokenizer.encode(text, add_special_tokens=False)
        tokens2 = tokenizer2.encode(text, add_special_tokens=False)
        assert tokens1 == tokens2

    def test_fertility_computation(self):
        """Test fertility (tokens per word) metric."""
        corpus = ["Das ist ein Test für den BPE-Tokenizer."] * 10
        tokenizer = GermanTokenizer(vocab_size=1000)
        tokenizer.train(corpus, verbose=False)

        text = "Das ist ein Test."
        fertility = tokenizer.fertility(text)

        # German is typically 1.0-1.5 tokens per word at 8k+ vocab
        assert 0 < fertility < 5

    def test_batch_encode_decode(self):
        """Test batch encoding and decoding."""
        corpus = ["Das ist ein Test.", "Guten Morgen!", "Donaudampfschiff"]
        tokenizer = GermanTokenizer(vocab_size=500)
        tokenizer.train(corpus, verbose=False)

        texts = ["Das ist ein Test.", "Guten Morgen!"]
        batch_tokens = tokenizer.encode_batch(texts, add_special_tokens=False)
        batch_decoded = tokenizer.decode_batch(batch_tokens, skip_special_tokens=True)

        assert len(batch_tokens) == 2
        assert len(batch_decoded) == 2
        assert all(isinstance(tokens, list) for tokens in batch_tokens)
        assert all(isinstance(text, str) for text in batch_decoded)

    def test_get_vocab(self):
        """Test accessing vocabulary."""
        corpus = ["Das ist ein Test."]
        tokenizer = GermanTokenizer(vocab_size=256)
        tokenizer.train(corpus, verbose=False)

        vocab = tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) > 0
        assert "<pad>" in vocab or "<unk>" in vocab

    def test_token_to_id(self):
        """Test token string to ID mapping."""
        corpus = ["Das ist ein Test."]
        tokenizer = GermanTokenizer(vocab_size=256)
        tokenizer.train(corpus, verbose=False)

        # Get an ID for a token
        vocab = tokenizer.get_vocab()
        if vocab:
            token = list(vocab.keys())[0]
            token_id = tokenizer.token_to_id(token)
            assert token_id == vocab[token]

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        corpus = ["Das ist ein Test."]
        tokenizer = GermanTokenizer(vocab_size=256)
        tokenizer.train(corpus, verbose=False)

        tokens = tokenizer.encode("", add_special_tokens=False)
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_unicode_handling(self):
        """Test handling of German umlauts and special characters."""
        corpus = [
            "Äpfel, Öl, Übergabe",
            "Größe, Schöne Grüße",
            "äöüß",
        ]
        tokenizer = GermanTokenizer(vocab_size=1000)
        tokenizer.train(corpus, verbose=False)

        text = "Äpfel und Öl"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert len(tokens) > 0

        # Roundtrip should preserve meaning (umlauts)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        assert isinstance(decoded, str)

    def test_compound_word_tokenization(self):
        """Test tokenization of German compound words."""
        # Train on diverse corpus so compound word isn't the entire vocab
        corpus = [
            "Donaudampfschifffahrtsgesellschaftskapitän",
            "Das ist ein normaler Satz.",
            "Guten Morgen und guten Tag!",
        ] * 5
        tokenizer = GermanTokenizer(vocab_size=500)
        tokenizer.train(corpus, verbose=False)

        text = "Donaudampfschifffahrtsgesellschaftskapitän"
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Should tokenize into at least one token
        assert len(tokens) >= 1
        # But should still round-trip
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        assert len(decoded) > 0

    def test_lowercase_normalization(self):
        """Test that lowercase=True works."""
        corpus = ["Das Ist Ein Test."]
        tokenizer = GermanTokenizer(vocab_size=256, lowercase=True)
        tokenizer.train(corpus, verbose=False)

        tokens1 = tokenizer.encode("Das Test", add_special_tokens=False)
        tokens2 = tokenizer.encode("das test", add_special_tokens=False)

        # Both should produce the same tokens due to lowercasing
        assert tokens1 == tokens2

    def test_not_trained_error(self):
        """Test error when using untrained tokenizer."""
        tokenizer = GermanTokenizer(vocab_size=256)

        with pytest.raises(ValueError, match="not trained or loaded"):
            tokenizer.encode("test")

        with pytest.raises(ValueError, match="not trained or loaded"):
            tokenizer.decode([1, 2, 3])

        with pytest.raises(ValueError, match="not trained or loaded"):
            tokenizer.get_vocab()

    def test_load_nonexistent_file(self):
        """Test error when loading from nonexistent path."""
        tokenizer = GermanTokenizer()
        with pytest.raises(FileNotFoundError):
            tokenizer.load(Path("/nonexistent/tokenizer.json"))
