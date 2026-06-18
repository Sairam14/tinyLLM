"""
Production-grade German LLM using Pre-LayerNorm, Flash Attention, and KV-cache.

Key differences from educational main.py:
- Pre-LayerNorm instead of Post-LN for training stability
- F.scaled_dot_product_attention with is_causal=True (dispatches to Flash Attention 2 on V100/A100)
- GELU instead of ReLU for FFN
- Dropout for regularization
- Weight tying (embedding ↔ lm_head)
- KV-cache support for efficient inference
- Final LayerNorm before lm_head
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyllm.config import ModelConfig


@dataclass
class TransformerOutput:
    """Output from transformer model."""

    logits: torch.Tensor  # [batch, seq_len, vocab_size]
    present_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None  # KV cache
    hidden_states: Optional[torch.Tensor] = None  # [batch, seq_len, d_model]


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for efficient long-context attention."""

    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply rotary embeddings to query/key.

        Args:
            x: [batch, n_heads, seq_len, d_k]
            seq_len: Optional custom sequence length (for KV-cache)

        Returns:
            x with rotary embeddings applied
        """
        seq_len = seq_len or x.size(2)
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, d_model/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, d_model]
        cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, d_model]
        sin = emb.sin()[None, None, :, :]

        # Apply rotation to x
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        x_rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rot * sin


class TransformerBlock(nn.Module):
    """Single transformer block: MHA → FFN with Pre-LN and residual connections."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Pre-LN: normalize before sublayer
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)

        # Multi-head attention
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with Pre-LN and optional KV-cache.

        Args:
            x: [batch, seq_len, d_model]
            past_kv: Optional (K, V) from previous positions: (K: [batch, n_heads, past_len, d_k], V: [batch, n_heads, past_len, d_k])
            is_causal: Apply causal mask (True for training/non-cached inference, False for KV-cache)

        Returns:
            output: [batch, seq_len, d_model]
            present_kv: (K, V) for current position
        """
        batch_size, seq_len, d_model = x.shape
        n_heads = self.config.n_heads
        d_k = self.config.d_k

        # Pre-LN: normalize then apply attention
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)

        # Concatenate with past KV if available (KV-cache)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # Concat on seq dimension
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k.detach(), v.detach())

        # Flash Attention (PyTorch 2.0+): automatic kernel fusion
        # When is_causal=True with full K/V, applies causal mask
        # When is_causal=False with cached K/V, no masking (cache is already ordered)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0.0, is_causal=is_causal
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_out = self.out_proj(attn_out)
        attn_out = self.attn_dropout(attn_out)

        # Residual connection
        x = x + attn_out

        # Pre-LN: normalize then apply FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        ffn_out = self.ffn_dropout(ffn_out)

        # Residual connection
        x = x + ffn_out

        return x, present_kv


class GermanLM(nn.Module):
    """Production German language model with Pre-LN, Flash Attention, KV-cache support."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embeddings (sinusoidal)
        self.register_buffer(
            "position_embedding",
            self._get_sinusoidal_encoding(config.max_seq_len, config.d_model),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final layer norm (required with Pre-LN)
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _get_sinusoidal_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Compute sinusoidal positional encoding."""
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        dim_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
        angle_rates = 1 / (10000 ** (dim_indices / d_model))

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * angle_rates)
        pe[:, 1::2] = torch.cos(position * angle_rates)
        return pe

    def _init_weights(self, module: nn.Module):
        """Initialize weights using standard transformer initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_kv: bool = False,
    ) -> TransformerOutput:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len]
            past_kv: Optional list of (K, V) tensors from previous positions for KV-cache
            return_kv: If True, return KV-cache for next iteration

        Returns:
            TransformerOutput with logits and optionally KV-cache
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]

        # Add positional embeddings
        pos_ids = torch.arange(seq_len, device=device)
        x = x + self.position_embedding[pos_ids].unsqueeze(0)

        # Apply transformer blocks with optional KV-cache
        present_kv_list = []
        is_causal = past_kv is None  # Only apply causal mask if not using KV-cache

        for i, block in enumerate(self.blocks):
            block_past_kv = past_kv[i] if past_kv is not None else None
            x, present_kv = block(x, past_kv=block_past_kv, is_causal=is_causal)
            if return_kv:
                present_kv_list.append(present_kv)

        # Final layer norm (Pre-LN requires this)
        x = self.final_norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return TransformerOutput(
            logits=logits,
            present_kv=present_kv_list if return_kv else None,
            hidden_states=x,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV-cache.

        Args:
            input_ids: [batch, seq_len] initial prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus (top-p) sampling

        Returns:
            [batch, seq_len + max_new_tokens]
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        generated_ids = input_ids.clone()

        # Initial forward pass (build KV-cache)
        with torch.no_grad():
            output = self.forward(input_ids, return_kv=True)
            past_kv = output.present_kv

            for _ in range(max_new_tokens):
                logits = output.logits[:, -1, :]  # [batch, vocab_size]

                # Apply temperature
                logits = logits / temperature

                # Top-k filtering
                if top_k is not None:
                    k_logits, k_indices = torch.topk(logits, top_k, dim=-1)
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(-1, k_indices, k_logits)

                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 0] = False
                    logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

                # Softmax and sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

                # Append to sequence and generate next
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Forward pass with KV-cache
                output = self.forward(next_token, past_kv=past_kv, return_kv=True)
                past_kv = output.present_kv

        return generated_ids
