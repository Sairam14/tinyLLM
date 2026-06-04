# tinyLLM — BPE Tokenizer for German

A lightweight implementation of Byte Pair Encoding (BPE) tokenization focused on understanding how language structure affects token efficiency. This project trains and analyzes a German tokenizer to compare vocabulary fertility between German and English.

## Project Goal

Demonstrate that German's compound word structure produces more semantically dense tokens than English, requiring fewer tokens to express equivalent meaning. The analysis compares:

- **German compounds** (e.g., "Donaudampfschifffahrtsgesellschaft") → fewer tokens
- **English phrases** (e.g., "Danube steamship company") → more tokens

This is particularly relevant for LLM training: German models need ~2× the vocabulary size to achieve the same token efficiency as English models.

## Files

- **`bpe_tokenizer.py`** — Core BPE tokenizer implementation
  - `BPETokeniser` class: train, encode/decode, calculate fertility metrics
  - Supports custom vocabulary sizes and compound word analysis
  
- **`train_tokenizer.py`** — Training script and experiments
  - Uses German corpus from Project Gutenberg (Kafka, Goethe, Schiller)
  - Trains tokenizer with configurable vocab sizes (2k–32k)
  - Compares fertility across German compounds vs English equivalents
  - Generates graphs 

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

### Train and Analyze

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

### Example Output

```
FERTILITY ANALYSIS — German compounds vs English equivalents
Donaudampfschifffahrtsgesellschaft       14 tokens      3 words
→ [47, 122, 85, 201, 5, ...]

Total                                    42 tokens     39 words
German is 1.08× more token-dense than English equivalents
```

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

## Next Steps

Part 2 will implement multi-head attention with this tokenizer to build a tiny transformer for German text generation.

## License

MIT — Educational and research use
