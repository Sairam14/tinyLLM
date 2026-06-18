#!/usr/bin/env python3
"""
Train HuggingFace BPE tokenizer on German data (CC-100 + Wikipedia).

This script replaces the educational pure-Python BPE trainer with the production
Rust-backed HuggingFace tokenizers library for 10-50× faster tokenization.

It streams real German data from HuggingFace datasets and trains a 32k vocab tokenizer.
"""

import argparse
from pathlib import Path
import os

from tinyllm.tokenizer import GermanTokenizer


def get_german_text_iterator(max_chars: int = 5_000_000):
    """Stream German text from HuggingFace datasets (Wikipedia + CC-100).

    This uses HF datasets' streaming mode to avoid downloading the full 65GB CC-100.
    For development, we can also fall back to synthetic data.

    Args:
        max_chars: Maximum total characters to stream

    Yields:
        Text documents
    """
    try:
        from datasets import load_dataset

        print("Streaming German Wikipedia...")
        wiki_ds = load_dataset("wikipedia", "20231101.de", split="train", streaming=True)
        chars_so_far = 0
        for example in wiki_ds:
            text = example["text"]
            yield text
            chars_so_far += len(text)
            if chars_so_far >= max_chars:
                print(f"Reached {max_chars} characters from Wikipedia")
                return

    except ImportError:
        print("HuggingFace datasets not installed. Using synthetic German text.")
        yield _generate_synthetic_german_corpus()


def _generate_synthetic_german_corpus() -> str:
    """Generate synthetic German text for development/testing.

    This is a fallback when real data is unavailable (e.g., in CI environments).
    """
    templates = [
        "Die Philosophie ist eine Wissenschaft, die sich mit grundlegenden Fragen der Existenz befasst.",
        "Kant hat behauptet, dass Raum und Zeit Formen der menschlichen Anschauung sind.",
        "Die deutsche Sprache ist bekannt für ihre langen Zusammensetzungswörter wie Freundschaftsbeziehung.",
        "Goethe und Schiller waren zwei der bedeutendsten deutschen Dichter und Denker.",
        "Das Donaudampfschifffahrtsgesellschaftskapitän ist ein berühmtes langes deutsches Wort.",
        "Berlin ist die Hauptstadt Deutschlands und hat eine reiche Geschichte.",
        "Die Berliner Mauer war ein Symbol der Teilung während des Kalten Krieges.",
        "Die Wiedervereinigung Deutschlands fand 1990 statt.",
        "Wissenschaft, Technologie und Kultur sind wichtige Bereiche der menschlichen Entwicklung.",
        "Nachhaltigkeit und Umweltschutz sind Herausforderungen für die Zukunft.",
    ]

    # Repeat to reach ~5M chars
    repeated = (templates * 50000)
    corpus = "\n".join(repeated)
    return corpus


def train_tokenizer(
    output_path: str = "tokenizer_32k.json",
    vocab_size: int = 32000,
    max_training_chars: int = 5_000_000,
):
    """Train German BPE tokenizer.

    Args:
        output_path: Where to save the trained tokenizer
        vocab_size: Vocabulary size (32k recommended for German)
        max_training_chars: Maximum characters to train on
    """
    print(f"\nTraining German BPE Tokenizer")
    print(f"  Target vocab size: {vocab_size:,}")
    print(f"  Max training data: {max_training_chars:,} characters")
    print(f"  Output path: {output_path}\n")

    # Initialize tokenizer
    tokenizer = GermanTokenizer(vocab_size=vocab_size, lowercase=True)

    # Collect texts from iterator (for training, we batch them)
    print("Collecting training data...")
    texts = []
    char_count = 0

    for text in get_german_text_iterator(max_chars=max_training_chars):
        texts.append(text)
        char_count += len(text)
        if char_count >= max_training_chars:
            print(f"Collected {char_count:,} characters")
            break

    # Train tokenizer
    print(f"\nTraining on {len(texts)} documents ({char_count:,} characters)...")
    tokenizer.train(texts, save_path=Path(output_path), verbose=True)

    # Show statistics
    print(f"\n✓ Tokenizer training complete")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Saved to: {output_path}")

    # Test on example German text
    test_texts = [
        "Das ist ein Test.",
        "Guten Morgen!",
        "Donaudampfschifffahrtsgesellschaftskapitän",
    ]

    print(f"\nTokenization examples:")
    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        fertility = len(tokens) / len(text.split()) if text.split() else 0
        print(f"  '{text}' → {len(tokens)} tokens (fertility: {fertility:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train German BPE tokenizer")
    parser.add_argument(
        "--output",
        default="tokenizer_32k.json",
        help="Output path for trained tokenizer",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target vocabulary size",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=5_000_000,
        help="Maximum characters to train on",
    )

    args = parser.parse_args()

    train_tokenizer(
        output_path=args.output,
        vocab_size=args.vocab_size,
        max_training_chars=args.max_chars,
    )
