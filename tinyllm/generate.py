"""
Inference utilities: KV-cache generation, sampling strategies, and batching.

KV-cache makes inference O(N) per token instead of O(N²), enabling fast streaming.
"""

from typing import Optional, List, Tuple, Callable
import torch
import torch.nn.functional as F

from tinyllm.model import GermanLM
from tinyllm.tokenizer import GermanTokenizer


class KVCacheGenerator:
    """Generate text using KV-cache for efficient inference."""

    def __init__(
        self,
        model: GermanLM,
        tokenizer: GermanTokenizer,
        device: str = "cuda",
        max_new_tokens: int = 100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: int = 3,
        do_sample: bool = True,
    ) -> str:
        """Generate text from a prompt using KV-cache.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate (uses default if None)
            temperature: Sampling temperature (0 = greedy)
            top_k: Keep top-k most likely tokens
            top_p: Keep tokens until cumulative probability exceeds p (nucleus sampling)
            eos_token_id: Token ID for end-of-sequence
            do_sample: Use sampling (True) or greedy (False)

        Returns:
            Generated text
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens

        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Initial forward pass (builds KV-cache)
        with torch.no_grad():
            output = self.model(input_ids, past_kv=None, return_kv=True)
            past_kv = output.present_kv

            generated_ids = input_ids.clone()

            for _ in range(max_new_tokens):
                # Get logits for last token
                logits = output.logits[:, -1, :]  # [batch, vocab_size]

                # Sample next token
                next_token_id = self._sample_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )

                # Check for EOS
                if next_token_id.item() == eos_token_id:
                    break

                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

                # Forward pass with KV-cache (only new token)
                output = self.model(next_token_id, past_kv=past_kv, return_kv=True)
                past_kv = output.present_kv

        # Decode and return
        generated_token_ids = generated_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return generated_text

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: int = 3,
    ) -> List[str]:
        """Generate text for a batch of prompts (careful with KV-cache memory!).

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            eos_token_id: EOS token ID

        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
            )
            results.append(result)
        return results

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Sample next token from logits.

        Args:
            logits: [batch, vocab_size] logits
            temperature: Temperature for sampling
            top_k: Top-k filtering
            top_p: Nucleus (top-p) filtering
            do_sample: Use sampling (True) or argmax (False)

        Returns:
            [batch, 1] sampled token IDs
        """
        if not do_sample or temperature == 0:
            # Greedy
            return logits.argmax(dim=-1, keepdim=True)

        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(-1, top_k_indices, top_k_logits)

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 0] = False  # Keep at least one token
            logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

        # Sample from filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token


class StreamingGenerator:
    """Generator that yields tokens one at a time (for streaming APIs)."""

    def __init__(
        self,
        model: GermanLM,
        tokenizer: GermanTokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """Yield generated tokens one at a time.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering

        Yields:
            Generated token strings
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            output = self.model(input_ids, past_kv=None, return_kv=True)
            past_kv = output.present_kv

            for _ in range(max_new_tokens):
                logits = output.logits[:, -1, :]

                # Sample
                next_token_id = KVCacheGenerator._sample_token(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p
                )

                # Decode and yield
                token_str = self.tokenizer.decode([next_token_id.item()], skip_special_tokens=False)
                yield token_str

                # Forward with cache
                output = self.model(next_token_id, past_kv=past_kv, return_kv=True)
                past_kv = output.present_kv
