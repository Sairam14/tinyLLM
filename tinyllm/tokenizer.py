"""
Production tokenizer wrapper around HuggingFace tokenizers library (Rust-backed).

Provides a clean interface matching the educational BPETokeniser while using
the high-performance Rust implementation from tokenizers.

This replaces the pure-Python BPETokeniser for production to avoid the O(merges × word_length)
bottleneck during data processing. The algorithm is identical; only the implementation language differs.
"""

from pathlib import Path
from typing import List, Union, Optional
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class GermanTokenizer:
    """Wrapper around HuggingFace tokenizers.Tokenizer for German text.

    Public interface matches BPETokeniser from the educational codebase.
    Uses HuggingFace's Rust-backed BPE (10-50× faster than pure Python).

    Special tokens: <pad>, <unk>, <bos>, <eos>
    """

    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, vocab_size: int = 32000, lowercase: bool = True):
        """Initialize tokenizer.

        Args:
            vocab_size: Target vocabulary size
            lowercase: Whether to lowercase text during normalization
        """
        self._target_vocab_size = vocab_size
        self.lowercase = lowercase
        self.tokenizer: Optional[Tokenizer] = None
        self._vocab_size_actual = vocab_size

    def train(
        self,
        texts: Union[str, List[str]],
        save_path: Optional[Path] = None,
        verbose: bool = True,
    ) -> None:
        """Train tokenizer on text corpus.

        Args:
            texts: Single text string or list of text strings
            save_path: Optional path to save trained tokenizer
            verbose: Print training progress
        """
        from tokenizers import decoders
        from tokenizers.trainers import BpeTrainer

        if isinstance(texts, str):
            texts = [texts]

        # Initialize BPE model
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))

        # Normalization: NFC (preserves umlauts), lowercase
        if self.lowercase:
            from tokenizers.normalizers import Lowercase

            tokenizer.normalizer = Sequence([NFC(), Lowercase()])
        else:
            tokenizer.normalizer = NFC()

        # Pre-tokenization: split on whitespace
        tokenizer.pre_tokenizer = Whitespace()

        # Train BPE
        trainer = BpeTrainer(
            vocab_size=self._target_vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            min_frequency=1,
            show_progress=verbose,
        )
        tokenizer.train_from_iterator(texts, trainer)

        # Post-processing: add <bos> and <eos>
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            pair="<bos> $A <eos> <bos> $B <eos>",
            special_tokens=[
                ("<bos>", self.BOS_ID),
                ("<eos>", self.EOS_ID),
            ],
        )

        # Decoder
        tokenizer.decoder = decoders.ByteLevel()

        self.tokenizer = tokenizer
        self._vocab_size_actual = len(tokenizer.get_vocab())

        if save_path:
            self.save(save_path)

        if verbose:
            print(f"✓ Tokenizer trained with vocab size: {self._vocab_size_actual}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Add <bos> at start and <eos> at end

        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded. Call train() or load() first.")

        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Remove special tokens from output

        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded. Call train() or load() first.")

        if skip_special_tokens:
            # Filter out special token IDs
            token_ids = [tid for tid in token_ids if tid not in (self.PAD_ID, self.BOS_ID, self.EOS_ID)]

        text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return text

    def save(self, path: Path) -> None:
        """Save tokenizer to file.

        Args:
            path: Output file path (JSON format)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    def load(self, path: Path) -> None:
        """Load tokenizer from file.

        Args:
            path: Path to tokenizer file (JSON format)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        self.tokenizer = Tokenizer.from_file(str(path))
        self._vocab_size_actual = len(self.tokenizer.get_vocab())

    @property
    def vocab_size(self) -> int:
        """Actual vocabulary size."""
        if self.tokenizer is None:
            return self._vocab_size_actual
        return len(self.tokenizer.get_vocab())

    def get_vocab(self) -> dict:
        """Get token to ID mapping.

        Returns:
            Dictionary mapping token strings to IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")
        return self.tokenizer.get_vocab()

    def get_id(self, token: str) -> Optional[int]:
        """Get ID for a specific token.

        Args:
            token: Token string

        Returns:
            Token ID or None if not in vocabulary
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")

        vocab = self.tokenizer.get_vocab()
        return vocab.get(token)

    def fertility(self, text: str) -> float:
        """Compute average tokens per word (lower is better).

        Args:
            text: Input text

        Returns:
            tokens / words ratio
        """
        tokens = self.encode(text, add_special_tokens=False)
        words = text.split()
        if len(words) == 0:
            return 0.0
        return len(tokens) / len(words)

    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Encode batch of texts.

        Args:
            texts: List of text strings
            add_special_tokens: Add special tokens

        Returns:
            List of token ID lists
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")

        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]

    def decode_batch(self, batch_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode batch of token ID sequences.

        Args:
            batch_ids: List of token ID lists
            skip_special_tokens: Remove special tokens

        Returns:
            List of decoded texts
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")

        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def token_to_id(self, token: str) -> Optional[int]:
        """Map token string to ID. Alias for get_id()."""
        return self.get_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Map ID to token string."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")

        vocab = self.tokenizer.get_vocab()
        # Reverse lookup
        for token, tid in vocab.items():
            if tid == token_id:
                return token
        return None
