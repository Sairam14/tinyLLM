"""
Data pipeline for training: streaming datasets, document packing, batching.

Uses HuggingFace datasets in streaming mode to avoid downloading gigabytes of data.
Implements document packing to maximize GPU utilization: ~100% vs ~60% with per-doc padding.
"""

from typing import Iterator, List, Tuple, Optional
import torch
from torch.utils.data import IterableDataset

from tinyllm.tokenizer import GermanTokenizer


class PackedDocumentDataset(IterableDataset):
    """IterableDataset that packs documents together for training.

    Instead of padding each document to max_seq_len (wasting tokens), we concatenate
    documents with <eos><bos> separators and slice into fixed-length chunks.
    This gives ~100% GPU utilization vs ~60% with per-doc padding.

    Args:
        texts: Iterator of text documents
        tokenizer: GermanTokenizer instance
        max_seq_len: Target sequence length for each chunk
        eos_id: ID of <eos> token
        bos_id: ID of <bos> token
    """

    def __init__(
        self,
        texts: Iterator[str],
        tokenizer: GermanTokenizer,
        max_seq_len: int = 2048,
        eos_id: int = 3,
        bos_id: int = 2,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_id = eos_id
        self.bos_id = bos_id

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield packed sequences with input_ids and labels.

        Returns:
            (input_ids, labels) both [max_seq_len] tensors
        """
        token_buffer = []

        for text in self.texts:
            # Tokenize document (without special tokens — we add them ourselves)
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Add separator tokens between documents
            if token_buffer:
                token_buffer.append(self.eos_id)
                token_buffer.append(self.bos_id)

            token_buffer.extend(tokens)

            # Yield full chunks
            while len(token_buffer) >= self.max_seq_len:
                chunk = token_buffer[: self.max_seq_len]
                token_buffer = token_buffer[self.max_seq_len :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)  # Everything except last token
                labels = torch.tensor(chunk[1:], dtype=torch.long)  # Everything except first token

                # Pad if needed (shouldn't happen often with packing)
                if len(input_ids) < self.max_seq_len - 1:
                    pad_len = self.max_seq_len - 1 - len(input_ids)
                    input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
                    labels = torch.cat([labels, torch.zeros(pad_len, dtype=torch.long)])

                yield input_ids, labels


class HFStreamingDataset(IterableDataset):
    """IterableDataset wrapping HuggingFace streaming datasets.

    Streams data from HF without downloading fully. Useful for CC-100 (65GB) and Wikipedia.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        config: Optional[str] = None,
        text_field: str = "text",
        streaming: bool = True,
    ):
        """Initialize streaming dataset.

        Args:
            dataset_name: HF dataset name (e.g., "wikitext", "cc100")
            split: Dataset split (e.g., "train", "validation")
            config: Dataset configuration (e.g., "de" for CC-100 German)
            text_field: Field name containing text in each example
            streaming: Use streaming mode (True) or download (False)
        """
        try:
            from datasets import load_dataset

            self.dataset = load_dataset(
                dataset_name, config, split=split, streaming=streaming
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")

        self.text_field = text_field

    def __iter__(self) -> Iterator[str]:
        """Yield text documents."""
        for example in self.dataset:
            text = example.get(self.text_field, "")
            if text:
                yield text


class SimpleIterableDataset(IterableDataset):
    """Simple IterableDataset from a list of texts (for testing)."""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __iter__(self) -> Iterator[str]:
        """Yield texts."""
        for text in self.texts:
            yield text


def create_train_val_split(
    dataset_iterator: Iterator[str],
    train_ratio: float = 0.95,
) -> Tuple[Iterator[str], Iterator[str]]:
    """Split dataset into train/val.

    Note: For streaming datasets, this splits on-the-fly. For reproducibility,
    prefer pre-split datasets or use HF's built-in splits.

    Args:
        dataset_iterator: Text document iterator
        train_ratio: Fraction to use for training

    Returns:
        (train_texts, val_texts) iterators
    """
    train_buffer = []
    val_buffer = []

    for i, text in enumerate(dataset_iterator):
        if i % 100 < (train_ratio * 100):
            train_buffer.append(text)
        else:
            val_buffer.append(text)

        # Yield in batches to avoid holding everything in memory
        if len(train_buffer) >= 1000:
            for t in train_buffer:
                yield t
            train_buffer = []

    # Yield remaining
    for t in train_buffer + val_buffer:
        yield t


class DataCollator:
    """Collate function for batching.

    Handles variable-length sequences by padding to max length in batch.
    """

    def __init__(self, pad_id: int = 0, max_seq_len: int = 2048):
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> dict:
        """Collate batch of (input_ids, labels) pairs.

        Args:
            batch: List of (input_ids, labels) tuples

        Returns:
            Dict with 'input_ids' and 'labels' batched tensors
        """
        input_ids_list = [item[0] for item in batch]
        labels_list = [item[1] for item in batch]

        # Stack into batch (assumes all same length after packing)
        input_ids = torch.stack(input_ids_list, dim=0)
        labels = torch.stack(labels_list, dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
