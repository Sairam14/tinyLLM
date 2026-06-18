"""
Configuration dataclasses for tinyLLM model and training.

These frozen dataclasses centralize all hyperparameters and are serialized
alongside model checkpoints for reproducibility and version tracking.
"""

from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture hyperparameters."""

    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    max_seq_len: int = 2048
    dropout: float = 0.1
    ffn_mult: float = 4.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.vocab_size < 256:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be >= 256")

    @property
    def d_ff(self) -> int:
        """Feed-forward hidden dimension."""
        return int(self.d_model * self.ffn_mult)

    @property
    def d_k(self) -> int:
        """Per-head key/value dimension."""
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "ffn_mult": self.ffn_mult,
            "norm_eps": self.norm_eps,
            "tie_embeddings": self.tie_embeddings,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create from dictionary."""
        return cls(**config_dict)

    def save(self, path: Path):
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    # Learning rate
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    total_steps: int = 100000
    lr_min_ratio: float = 0.1

    # Batch and sequence
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 2048

    # Optimization
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    use_gradient_checkpointing: bool = True

    # Precision and device
    dtype: str = "bfloat16"  # Options: "float32", "float16", "bfloat16"
    device: str = "cuda"  # Options: "cuda", "cpu"

    # Checkpointing and logging
    checkpoint_every: int = 500
    log_every: int = 50
    eval_every: int = 1000
    eval_steps: int = 100
    checkpoint_dir: str = "checkpoints"
    keep_checkpoints: int = 3

    # Distributed training
    use_ddp: bool = True
    backend: str = "nccl"

    # Data and tokenizer
    tokenizer_path: str = "tokenizer_32k.json"
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    num_workers: int = 4

    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "tinyllm"
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be < total_steps")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"dtype must be one of float32/float16/bfloat16, got {self.dtype}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_len": self.max_seq_len,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "dtype": self.dtype,
            "device": self.device,
            "checkpoint_every": self.checkpoint_every,
            "log_every": self.log_every,
            "eval_every": self.eval_every,
            "eval_steps": self.eval_steps,
            "checkpoint_dir": self.checkpoint_dir,
            "keep_checkpoints": self.keep_checkpoints,
            "use_ddp": self.use_ddp,
            "backend": self.backend,
            "tokenizer_path": self.tokenizer_path,
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path,
            "num_workers": self.num_workers,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainConfig":
        """Create from dictionary."""
        return cls(**config_dict)

    def save(self, path: Path):
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
