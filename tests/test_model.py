"""Tests for GermanLM model architecture."""

import pytest
import torch
import torch.nn as nn

from tinyllm.config import ModelConfig
from tinyllm.model import GermanLM, TransformerBlock


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_config_creation(self):
        """Test creating a valid config."""
        config = ModelConfig(
            vocab_size=32000,
            d_model=512,
            n_heads=8,
            n_layers=12,
            max_seq_len=2048,
        )
        assert config.vocab_size == 32000
        assert config.d_model == 512
        assert config.d_ff == 512 * 4

    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError, match="d_model.*must be divisible"):
            ModelConfig(d_model=511, n_heads=8)

    def test_config_serialization(self, tmp_path):
        """Test config save/load."""
        config = ModelConfig(vocab_size=32000, d_model=512)
        path = tmp_path / "config.json"

        config.save(path)
        assert path.exists()

        loaded = ModelConfig.load(path)
        assert loaded.vocab_size == config.vocab_size
        assert loaded.d_model == config.d_model


class TestTransformerBlock:
    """Test TransformerBlock component."""

    def test_block_forward(self):
        """Test forward pass without KV-cache."""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=1)
        block = TransformerBlock(config)

        x = torch.randn(2, 8, 256)  # [batch, seq_len, d_model]
        out, present_kv = block(x, past_kv=None, is_causal=True)

        assert out.shape == x.shape
        assert present_kv[0].shape == (2, 4, 8, 64)  # [batch, n_heads, seq_len, d_k]

    def test_block_kv_cache(self):
        """Test forward pass with KV-cache."""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=1)
        block = TransformerBlock(config)

        # Initial forward pass
        x_init = torch.randn(2, 8, 256)
        out_init, kv_init = block(x_init, past_kv=None, is_causal=True)

        # Next token with cache
        x_next = torch.randn(2, 1, 256)
        out_next, kv_next = block(x_next, past_kv=kv_init, is_causal=False)

        assert out_next.shape == (2, 1, 256)
        # KV should grow in seq_len dimension
        assert kv_next[0].shape[2] == 9  # 8 + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
class TestGermanLM:
    """Test GermanLM model."""

    def test_model_creation(self):
        """Test model initialization."""
        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=4,
            n_layers=2,
            max_seq_len=512,
        )
        model = GermanLM(config)

        # Check weight tying
        assert model.lm_head.weight is model.token_embedding.weight

    def test_model_forward_training(self):
        """Test forward pass in training mode (full sequence, no KV-cache)."""
        config = ModelConfig(vocab_size=1000, d_model=256, n_heads=4, n_layers=2)
        model = GermanLM(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 16))  # [batch, seq_len]
        output = model(input_ids, past_kv=None, return_kv=False)

        assert output.logits.shape == (2, 16, 1000)
        assert output.present_kv is None

    def test_model_forward_with_kv_cache(self):
        """Test forward pass with KV-cache (inference mode)."""
        config = ModelConfig(vocab_size=1000, d_model=256, n_heads=4, n_layers=2)
        model = GermanLM(config)
        model.eval()

        # Initial forward pass to build cache
        input_ids = torch.randint(0, 1000, (2, 8))
        output = model(input_ids, return_kv=True)
        kv_cache = output.present_kv

        assert len(kv_cache) == 2  # 2 layers

        # Next token with cache
        next_ids = torch.randint(0, 1000, (2, 1))
        output_next = model(next_ids, past_kv=kv_cache, return_kv=True)

        assert output_next.logits.shape == (2, 1, 1000)

    def test_model_generate(self):
        """Test autoregressive generation."""
        config = ModelConfig(vocab_size=100, d_model=128, n_heads=4, n_layers=2)
        model = GermanLM(config)
        model.eval()

        prompt = torch.tensor([[1, 2, 3]])  # [batch=1, seq_len=3]
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=5, temperature=1.0)

        assert generated.shape == (1, 8)  # 3 + 5

    def test_weight_tying_gradient_flow(self):
        """Test that weight tying gradients flow correctly."""
        config = ModelConfig(vocab_size=100, d_model=128, n_heads=4, n_layers=1)
        model = GermanLM(config)

        input_ids = torch.randint(0, 100, (2, 8))
        output = model(input_ids)
        loss = output.logits.sum()
        loss.backward()

        # Both embedding and lm_head should have gradients (shared weights)
        assert model.token_embedding.weight.grad is not None
        assert model.lm_head.weight.grad is model.token_embedding.weight.grad

    def test_model_on_device(self):
        """Test model device placement."""
        config = ModelConfig(vocab_size=100, d_model=128, n_heads=4, n_layers=1)
        model = GermanLM(config)

        assert next(model.parameters()).device.type == "cpu"

        if torch.cuda.is_available():
            model = model.cuda()
            assert next(model.parameters()).device.type == "cuda"

    def test_model_parameter_count(self):
        """Test parameter counting."""
        config = ModelConfig(vocab_size=1000, d_model=256, n_heads=4, n_layers=2)
        model = GermanLM(config)

        total_params = sum(p.numel() for p in model.parameters())
        # Rough check: embedding (1000×256) + 2 layers × (attention+ffn) + lm_head
        # Should be ballpark of 300k-500k parameters
        assert 100_000 < total_params < 1_000_000

    def test_different_batch_sizes(self):
        """Test model handles variable batch sizes."""
        config = ModelConfig(vocab_size=100, d_model=128, n_heads=4, n_layers=1)
        model = GermanLM(config)
        model.eval()

        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, 100, (batch_size, 16))
            output = model(input_ids)
            assert output.logits.shape == (batch_size, 16, 100)

    def test_dropout_training_vs_eval(self):
        """Test dropout is disabled in eval mode."""
        config = ModelConfig(vocab_size=100, d_model=128, n_heads=4, n_layers=1, dropout=0.5)
        model = GermanLM(config)
        input_ids = torch.randint(0, 100, (2, 8))

        # Training mode: outputs should vary due to dropout
        model.train()
        with torch.no_grad():
            out1 = model(input_ids).logits
            out2 = model(input_ids).logits
        # With high dropout, outputs should differ (stochastic)
        # Note: this is probabilistic and could occasionally fail
        assert not torch.allclose(out1, out2, atol=0.1)

        # Eval mode: outputs should be deterministic
        model.eval()
        with torch.no_grad():
            out1 = model(input_ids).logits
            out2 = model(input_ids).logits
        assert torch.allclose(out1, out2)
