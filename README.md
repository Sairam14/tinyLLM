# tinyLLM — Building a Tiny Transformer from Scratch

A series of educational implementations demonstrating how to build a transformer model from first principles. We start with tokenization (Part 1) and move to multi-head attention (Part 2), building toward a full language model.

## Overview

- **Part 1: BPE Tokenizer** — Byte Pair Encoding implementation for German text
- **Part 2: Multi-Head Attention** — Scaled dot-product attention, multi-head mechanism, and positional encoding

## Project Goal

Demonstrate that German's compound word structure produces more semantically dense tokens than English, requiring fewer tokens to express equivalent meaning. The analysis compares:

- **German compounds** (e.g., "Donaudampfschifffahrtsgesellschaft") → fewer tokens
- **English phrases** (e.g., "Danube steamship company") → more tokens

This is particularly relevant for LLM training: German models need ~2× the vocabulary size to achieve the same token efficiency as English models.

## Files

### Part 1: Tokenization

- **`bpe_tokenizer.py`** — Core BPE tokenizer implementation
  - `BPETokeniser` class: train, encode/decode, calculate fertility metrics
  - Supports custom vocabulary sizes and compound word analysis
  
- **`train_tokenizer.py`** — Training script and experiments
  - Uses German corpus from Project Gutenberg (Kafka, Goethe, Schiller)
  - Trains tokenizer with configurable vocab sizes (2k–32k)
  - Compares fertility across German compounds vs English equivalents

### Part 2: Multi-Head Attention

- **`multi_head_attention.py`** — Core MHA implementation (~80 lines)
  - `PositionalEncoding`: Sine/cosine positional encoding
  - `ScaledDotProductAttention`: Core attention mechanism
  - `MultiHeadAttention`: Full multi-head attention module
  
- **`MHA_GUIDE.md`** — Complete conceptual guide
  - Explains Q/K/V (Query, Key, Value)
  - Scaled dot-product attention math
  - Why multiple heads work better
  - Positional encoding visualization
  - Parameter explanations and usage examples

- **`mha_examples.py`** — Advanced examples and patterns
  - Self-attention (encoder-style)
  - Cross-attention (encoder-decoder)
  - Causal attention (decoder-only, "can't look at future")
  - Transformer block with FFN and residual connections
  - Simple language model for next-token prediction
  - Attention head analysis and visualization

## Setup

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (fast Python package installer)

### Installation

```bash
# Create virtual environment
uv venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Or on Windows:
# .venv\Scripts\activate
```

No external dependencies required—the tokenizer uses only Python standard library.

## Usage

### Quick Demo

Run the included demo script to see the tokenizer in action:

```bash
python demo.py
```

This trains a tokenizer on sample German text and demonstrates:
- BPE training and merge process
- Encoding and decoding
- Fertility calculation (tokens per word)
- Compound word analysis

#### Demo Results

```
Training BPE tokeniser | target vocab size: 500
Base character vocabulary: 40 characters
Merges to perform: 456
No more pairs to merge after 169 merges.

Training complete. Final vocabulary size: 213

Encoding: Das ist ein Test.
Token IDs: [2, 58, 48, 59, 87, 6, 3]
Number of tokens: 7
Decoded: das ist ein test .

Fertility: 2.50 tokens/word

Word: Donaudampfschifffahrtsgesellschaftskapitän
Tokens (31): ['D', 'o', 'na', 'u', 'da', 'm', 'p', 'f', 'sch', 'i', 'ff', 'f', 'a', 'h', 'r', 'ts', 'ge', 's', 'e', 'l', 'l', 'sch', 'a', 'f', 'ts', 'k', 'a', 'p', 'i', 'tä', 'n</w>']
  IDs: [1, 26, 76, 34, 55, 23, 27, 15, 51, 20, 77, 15, 8, 18, 28, 78, 57, 30, 13, 22, 22, 51, 8, 15, 78, 21, 8, 27, 20, 79, 25]
```

### Full Training

For full training on a larger corpus:

```bash
python train_tokenizer.py
```

This will:
1. Use the ~5MB of German public domain text
2. Train a BPE tokenizer (vocab_size=8000)
3. Show sanity checks and decoded output
4. Run fertility comparison on compound words
5. Benchmark different vocab sizes (2k, 4k, 8k, 16k, 32k)
6. Save the trained tokenizer as `german_bpe_8k.json`

## Key Findings

- **German vocab efficiency**: Compound words allow a single token to represent multi-word English concepts
- **Optimal vocab size**: Fertility plateaus around 8k for German (vs 4k for English)
- **Training time**: ~2-5 seconds for 2M character corpus on standard hardware
- **Token density**: German saves ~7-8% token count vs English equivalents

## Metrics

### Fertility

Token fertility = (number of tokens) / (number of words)

- Lower fertility = more efficient tokenization
- German typically achieves 0.95–1.1× fertility vs English's 1.0–1.2×

### Vocabulary Size Trade-offs

| Vocab Size | Avg Fertility | Training Time |
|-----------|---------------|---------------|
| 2,000 | 1.542 | 0.8s |
| 4,000 | 1.203 | 1.1s |
| 8,000 | 1.085 | 1.5s |
| 16,000 | 1.068 | 2.1s |
| 32,000 | 1.065 | 3.2s |

## Data Sources

Training corpus sourced from [Project Gutenberg](https://www.gutenberg.org/) (public domain):
- Kafka — *Der Proceß* (7988)
- Goethe — *Faust* (2229)
- Schiller — *Die Räuber* (6784)

No API keys, authentication, or paid resources required.

## Quick Start for Part 2

```bash
# Run the multi-head attention demo
python multi_head_attention.py

# See 6 advanced examples (self-attention, cross-attention, causal, etc.)
python mha_examples.py

# Read the complete guide
cat MHA_GUIDE.md
```

## Next Steps

- Part 3: Combine tokenizer + MHA to build a full Transformer encoder
- Part 4: Train on next-token prediction for German text generation
- Part 5: Add GPT-style decoding and sampling strategies

## License

MIT — Educational and research use
