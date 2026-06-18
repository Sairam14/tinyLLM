#!/usr/bin/env python3
"""
Export model to HuggingFace format (PreTrainedModel), then convert to GGUF with llama.cpp.

GGUF is the standard format for llama.cpp (C++/C inference engine).
Enables on-premise CPU inference without GPU, with 4-bit quantization.

This script saves in HF format; then use llama.cpp's convert_hf_to_gguf.py:
    python llama.cpp/convert_hf_to_gguf.py tinyllm_hf/ --outfile model.gguf --outtype f16
    ./llama.cpp/llama-quantize model.gguf model_q4.gguf Q4_K_M
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from tinyllm.config import ModelConfig
from tinyllm.model import GermanLM


class GermanLMConfig:
    """HuggingFace-compatible config class for GermanLM.

    This mimics transformers.PretrainedConfig for compatibility with convert_hf_to_gguf.py.
    """

    def __init__(self, **kwargs):
        # Map tinyLLM config to HF format
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.hidden_size = kwargs.get("d_model", 512)
        self.num_attention_heads = kwargs.get("n_heads", 8)
        self.num_hidden_layers = kwargs.get("n_layers", 12)
        self.intermediate_size = int(self.hidden_size * kwargs.get("ffn_mult", 4.0))
        self.max_position_embeddings = kwargs.get("max_seq_len", 2048)
        self.hidden_dropout_prob = kwargs.get("dropout", 0.1)
        self.attention_probs_dropout_prob = kwargs.get("dropout", 0.1)
        self.initializer_range = 0.02
        self.layer_norm_eps = kwargs.get("norm_eps", 1e-6)
        self.tie_word_embeddings = kwargs.get("tie_embeddings", True)
        self.architectures = ["GermanLMForCausalLM"]
        self.model_type = "german-lm"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "architectures": self.architectures,
            "model_type": self.model_type,
        }

    @classmethod
    def from_pretrained(cls, path: Path) -> "GermanLMConfig":
        """Load from config.json."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def convert_to_hf_format(
    model: GermanLM,
    model_config: ModelConfig,
    output_dir: Path,
) -> None:
    """Convert tinyLLM model to HuggingFace format.

    Saves:
    - model weights (PyTorch)
    - config.json (HF format)
    - pytorch_model.bin or safe_tensors

    Args:
        model: GermanLM instance
        model_config: ModelConfig
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert config to HF format
    hf_config = GermanLMConfig(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout,
        ffn_mult=model_config.ffn_mult,
        norm_eps=model_config.norm_eps,
        tie_embeddings=model_config.tie_embeddings,
    )

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(hf_config.to_dict(), f, indent=2)
    print(f"✓ Saved config to {config_path}")

    # Save model weights
    state_dict = model.state_dict()
    weights_path = output_dir / "pytorch_model.bin"
    torch.save(state_dict, weights_path)
    print(f"✓ Saved weights to {weights_path}")

    # Save model.json (for convert_hf_to_gguf.py detection)
    model_json = {
        "architectures": ["GermanLMForCausalLM"],
        "model_type": "german-lm",
    }
    model_json_path = output_dir / "model.json"
    with open(model_json_path, "w") as f:
        json.dump(model_json, f, indent=2)

    print(f"\n✓ Converted to HuggingFace format in {output_dir}")
    print(f"Next steps:")
    print(f"  1. python llama.cpp/convert_hf_to_gguf.py {output_dir} --outfile model.gguf --outtype f16")
    print(f"  2. ./llama.cpp/llama-quantize model.gguf model_q4.gguf Q4_K_M")
    print(f"  3. ./llama.cpp/llama-cli -m model_q4.gguf -p 'Guten Morgen'")


def main():
    """Export checkpoint to GGUF-compatible format."""
    parser = argparse.ArgumentParser(description="Export GermanLM to HuggingFace format (GGUF-compatible)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output-dir", default="tinyllm_hf", help="Output directory for HF format")

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    model = GermanLM(model_config)
    model.load_state_dict(checkpoint["model"])

    # Convert and save
    convert_to_hf_format(model, model_config, Path(args.output_dir))


if __name__ == "__main__":
    main()
