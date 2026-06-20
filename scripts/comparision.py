"""
Comparison script: BPE vs Unigram vs morphology-aware pre-segmentation
on the SAME German corpus used in Part 1.

Run this on your local machine (not in a sandboxed environment) where you
have already downloaded the Kafka/Goethe/Schiller corpus from Part 1's
train_tokenizer.py -- this script re-uses that exact corpus so the Part 4
numbers are directly comparable to Part 1's BPE-only results.

Requirements:
    pip install sentencepiece tokenizers

Usage:
    python scripts/comparison.py
    or
    cd scripts && python comparison.py
"""

import os
import sys
import time
import re
from collections import Counter

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────────────────
# Step 0: Load the SAME corpus from Part 1
# ─────────────────────────────────────────────────────────────────────────

def load_part1_corpus(max_chars: int = 2_000_000) -> str:
    """
    Re-uses the exact same download logic from Part 1's train_tokenizer.py.
    If the files are already downloaded locally from running Part 1, this
    will reuse them rather than re-downloading.
    """
    import urllib.request

    urls = [
        "https://www.gutenberg.org/cache/epub/7988/pg7988.txt",   # Kafka - Der Proceß
        "https://www.gutenberg.org/cache/epub/2229/pg2229.txt",   # Goethe - Faust
        "https://www.gutenberg.org/cache/epub/6784/pg6784.txt",   # Schiller - Die Räuber
    ]

    combined = []
    total_chars = 0

    for url in urls:
        filename = url.split('/')[-1]
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)

        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        combined.append(text)
        total_chars += len(text)
        print(f"  {filename}: {len(text):,} chars")

        if total_chars >= max_chars:
            break

    corpus = '\n'.join(combined)[:max_chars]
    print(f"\nTotal corpus: {len(corpus):,} characters\n")
    return corpus


# ─────────────────────────────────────────────────────────────────────────
# Step 1: BPE -- your Part 1 tokeniser (re-imported, not reimplemented)
# ─────────────────────────────────────────────────────────────────────────

def run_bpe(corpus: str, vocab_size: int = 8000):
    """
    Uses your actual Part 1 BPETokeniser class.
    Make sure bpe_tokenizer.py from Part 1 is on the same path.
    """
    from bpe_tokenizer import BPETokeniser

    start = time.time()
    tok = BPETokeniser(vocab_size=vocab_size)
    tok.train(corpus, verbose=False)
    elapsed = time.time() - start

    return tok, elapsed


# ─────────────────────────────────────────────────────────────────────────
# Step 2: Unigram via SentencePiece -- the genuinely different algorithm
# ─────────────────────────────────────────────────────────────────────────

def run_unigram(corpus: str, vocab_size: int = 8000):
    """
    Trains a Unigram language model tokeniser using SentencePiece --
    the same library used by T5, mT5, ALBERT, XLNet.

    Unigram selects tokens by maximising corpus likelihood via EM,
    rather than BPE's greedy frequency-based merging. This is expected
    to produce different (often better) segmentation of compound words,
    because it can represent ambiguous segmentations probabilistically
    during training (subword regularisation) rather than committing to
    one deterministic split.
    """
    import sentencepiece as spm

    # SentencePiece trains from a file, not a string
    corpus_path = "_temp_corpus_for_unigram.txt"
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write(corpus)

    model_prefix = "_temp_unigram_model"

    start = time.time()
    try:
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='unigram',     # <-- the key difference from BPE
            character_coverage=1.0,    # cover all characters in the German corpus
            normalization_rule_name='nmt_nfkc',  # standard normalisation
        )
    except RuntimeError as e:
        # SentencePiece refuses to train if the corpus is too small to
        # support the requested vocab_size -- this happens on small test
        # corpora. On the real, multi-MB Part 1 corpus this should not
        # trigger at vocab_size=8000. If it does, the error message tells
        # you the maximum size it WILL support -- worth reporting honestly
        # in the article rather than silently lowering vocab_size to match.
        print(f"\n  SentencePiece could not train at vocab_size={vocab_size}: {e}")
        print(f"  This corpus may be too small for this vocab size.\n")
        raise
    elapsed = time.time() - start

    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    os.remove(corpus_path)

    return sp, elapsed


# ─────────────────────────────────────────────────────────────────────────
# Step 3: Morphology-aware pre-segmentation (simplified, rule-based)
# ─────────────────────────────────────────────────────────────────────────

# A small set of known German morpheme boundaries -- NOT a full morphological
# analyser (a real one, e.g. SMOR or Zmorge, requires a dedicated lexicon and
# rule engine). This is a simplified demonstration of the PRINCIPLE: split at
# linguistically plausible compound boundaries before running BPE, rather than
# relying purely on statistical frequency.

KNOWN_MORPHEMES = [
    "schiff", "fahrt", "gesellschaft", "dampf", "donau",
    "bundes", "verfassung", "gericht", "versicherung",
    "kraftfahrzeug", "haftpflicht", "regierung", "recht",
    "schutz", "gesundheit", "versorgung", "wohnung", "bau",
    "eisenbahn", "wirtschaft", "wissenschaft", "universitat",
    "stadt", "verwaltung", "wasser", "aufbereitung", "anlage",
    "klima", "wandel", "lebensmittel", "handel", "vereinbarung",
]
KNOWN_MORPHEMES.sort(key=len, reverse=True)  # longest match first


def morphology_presegment(word: str) -> list:
    """
    Greedily split a word at known morpheme boundaries, longest match first.
    This is a simplified stand-in for a real morphological analyser (e.g.
    SMOR for German) -- it demonstrates the PRINCIPLE that linguistic
    boundaries differ from statistically-discovered BPE merge points,
    not a production-grade morphological segmenter.
    """
    word_lower = word.lower()
    segments = []
    i = 0
    while i < len(word_lower):
        matched = False
        for morpheme in KNOWN_MORPHEMES:
            if word_lower[i:i+len(morpheme)] == morpheme:
                segments.append(word_lower[i:i+len(morpheme)])
                i += len(morpheme)
                matched = True
                break
        if not matched:
            # no known morpheme matches here -- take one character
            # and let BPE handle the remainder statistically
            segments.append(word_lower[i])
            i += 1
    return segments


def run_morphology_aware_bpe(corpus: str, vocab_size: int = 8000):
    """
    Pre-segments compound words at known morpheme boundaries, joins
    segments with a boundary marker, then trains BPE on the pre-segmented
    text. This lets BPE operate on linguistically-motivated units rather
    than raw characters for the words where morphemes are known.
    """
    from bpe_tokenizer import BPETokeniser

    # Pre-segment: for each word, if morphology_presegment finds known
    # morphemes, join them with a space so BPE's pre-tokeniser treats
    # them as separate "words" -- preserving the morpheme boundary
    # rather than letting BPE merge across it based on frequency alone.
    words = re.findall(r"[a-zA-ZäöüÄÖÜß]+|[^\sa-zA-ZäöüÄÖÜß]", corpus)
    presegmented_words = []
    for w in words:
        if re.match(r"[a-zA-ZäöüÄÖÜß]+", w) and len(w) > 12:
            # only pre-segment longer words -- short words rarely benefit
            segments = morphology_presegment(w)
            presegmented_words.append(' '.join(segments))
        else:
            presegmented_words.append(w)

    presegmented_corpus = ' '.join(presegmented_words)

    start = time.time()
    tok = BPETokeniser(vocab_size=vocab_size)
    tok.train(presegmented_corpus, verbose=False)
    elapsed = time.time() - start

    return tok, elapsed


# ─────────────────────────────────────────────────────────────────────────
# Step 4: The comparison -- same test words across all three approaches
# ─────────────────────────────────────────────────────────────────────────

TEST_COMPOUNDS = [
    "Donaudampfschifffahrtsgesellschaftskapitän",
    "Bundesverfassungsgericht",
    "Kraftfahrzeughaftpflichtversicherung",
    "Rechtsschutzversicherungsgesellschaftsangestellter",
    "Gesundheitsversorgungsreform",
]

TEST_SENTENCES = [
    "Die Bundesregierung hat das Gesetz verabschiedet.",
    "Das Kraftfahrzeug wurde ordnungsgemäß zugelassen.",
    "Die Donaudampfschifffahrtsgesellschaft transportierte Waren.",
]


def count_bpe_tokens(tok, word: str) -> int:
    return len(tok._encode_word(word.lower()))


def count_unigram_tokens(sp, word: str) -> int:
    return len(sp.encode(word, out_type=str))


def fertility_bpe(tok, sentences) -> float:
    total_words = sum(len(s.split()) for s in sentences)
    total_tokens = sum(
        sum(count_bpe_tokens(tok, w) for w in s.split())
        for s in sentences
    )
    return total_tokens / total_words


def fertility_unigram(sp, sentences) -> float:
    total_words = sum(len(s.split()) for s in sentences)
    total_tokens = sum(len(sp.encode(s, out_type=str)) for s in sentences)
    return total_tokens / total_words


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("PART 4: BPE vs Unigram vs Morphology-aware pre-segmentation")
    print("Same German corpus, same vocab size, same test words")
    print("=" * 70)

    corpus = load_part1_corpus(max_chars=2_000_000)
    VOCAB_SIZE = 8000

    print(f"\nTraining all three tokenisers at vocab_size={VOCAB_SIZE}...\n")

    print("[1/3] Training BPE (Part 1 baseline)...")
    bpe_tok, bpe_time = run_bpe(corpus, VOCAB_SIZE)
    print(f"      Done in {bpe_time:.1f}s")

    print("[2/3] Training Unigram (SentencePiece)...")
    unigram_tok, unigram_time = run_unigram(corpus, VOCAB_SIZE)
    print(f"      Done in {unigram_time:.1f}s")

    print("[3/3] Training morphology-aware BPE...")
    morph_tok, morph_time = run_morphology_aware_bpe(corpus, VOCAB_SIZE)
    print(f"      Done in {morph_time:.1f}s")

    print("\n" + "=" * 70)
    print("COMPOUND WORD TOKEN COUNTS")
    print("=" * 70)
    print(f"{'Word':<55} {'BPE':>6} {'Unigram':>8} {'Morph':>6}")
    print("-" * 70)

    for word in TEST_COMPOUNDS:
        bpe_n = count_bpe_tokens(bpe_tok, word)
        uni_n = count_unigram_tokens(unigram_tok, word)
        morph_n = count_bpe_tokens(morph_tok, word)  # uses pre-segmented BPE
        print(f"{word[:53]:<55} {bpe_n:>6} {uni_n:>8} {morph_n:>6}")

    print("\n" + "=" * 70)
    print("FERTILITY COMPARISON (tokens per word, test sentences)")
    print("=" * 70)

    bpe_fert = fertility_bpe(bpe_tok, TEST_SENTENCES)
    uni_fert = fertility_unigram(unigram_tok, TEST_SENTENCES)
    morph_fert = fertility_bpe(morph_tok, TEST_SENTENCES)

    print(f"BPE (Part 1 baseline):        {bpe_fert:.3f} tokens/word")
    print(f"Unigram (SentencePiece):      {uni_fert:.3f} tokens/word")
    print(f"Morphology-aware + BPE:       {morph_fert:.3f} tokens/word")

    print("\n" + "=" * 70)
    print("TRAINING TIME COMPARISON")
    print("=" * 70)
    print(f"BPE:               {bpe_time:.1f}s")
    print(f"Unigram:           {unigram_time:.1f}s")
    print(f"Morphology-aware:  {morph_time:.1f}s")

    print("\nThese are the REAL numbers for Part 4. Report them exactly as")
    print("measured -- do not round up or adjust to match a hypothesis.")
    print("If Unigram does NOT clearly beat BPE on this corpus size, that")
    print("is itself an honest and interesting finding worth reporting.")
