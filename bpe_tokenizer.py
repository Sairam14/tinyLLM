# bpe_tokenizer.py
# BPE tokeniser from scratch — no HuggingFace, no SentencePiece
# Part 1 of building a tiny German LLM from scratch
# Sairam Sundaram — github.com/Sairam14

import re
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import unicodedata


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Text normalisation
# German-specific: handle umlauts (ä ö ü Ä Ö Ü ß), smart quotes, em-dashes
# ─────────────────────────────────────────────────────────────────────────────

def normalise_german(text: str) -> str:
    """
    Normalise German text before tokenisation.
    
    Key decisions:
    - Keep umlauts as-is (do NOT convert ä→ae). Converting collapses
      distinct words: 'schon' (already) vs 'schön' (beautiful).
    - Lowercase everything — German nouns are capitalised but we want
      'Haus' and 'haus' to share the same token.
    - Strip unicode control characters but preserve standard punctuation.
    - Normalise to NFC (composed form) — ä can be stored as one
      codepoint U+00E4 or two (a + combining diaeresis). NFC ensures
      the former, so our character vocabulary is minimal.
    """
    # NFC normalisation — critical for German umlauts
    text = unicodedata.normalize('NFC', text)
    # Lowercase
    text = text.lower()
    # Normalise whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Pre-tokenisation
# Split text into words before BPE. 
# We use a pattern that respects German compound word boundaries.
# ─────────────────────────────────────────────────────────────────────────────

# GPT-2 style pre-tokenisation pattern — splits on:
# - Contractions (won't, I'm)
# - Punctuation
# - Whitespace-separated words
# The </w> suffix marks end-of-word — critical for BPE to learn
# that "ing" at word-end is different from "ing" inside a word.

PRETOK_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|[a-zA-ZäöüÄÖÜß]+|[0-9]+|[^\s\w]""",
    re.UNICODE
)

def pretokenise(text: str) -> List[str]:
    """
    Split text into words/tokens before applying BPE.
    Each word is converted to a tuple of characters with </w> appended
    to the last character to mark word boundaries.
    
    Example: "Haus" → ['H', 'a', 'u', 's</w>']
    
    Why word boundary markers matter for German:
    The suffix "-ung" inside "Bildung" (education) should eventually
    merge into a single token. Without </w>, "ung" at word-end and
    "ung" mid-word would be treated identically — which loses the
    morphological signal that German suffixes carry.
    """
    words = PRETOK_PATTERN.findall(text)
    return words


def word_to_char_sequence(word: str) -> Tuple[str, ...]:
    """Convert a word string to a tuple of characters with end-of-word marker."""
    if len(word) == 0:
        return tuple()
    chars = list(word)
    chars[-1] = chars[-1] + '</w>'
    return tuple(chars)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build initial vocabulary from corpus
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab_from_corpus(corpus: str) -> Dict[Tuple[str, ...], int]:
    """
    Tokenise the corpus at character level and count word frequencies.
    
    Returns a dict mapping character-tuple representations of words
    to their frequency in the corpus.
    
    Example output entry: ('H', 'a', 'u', 's</w>') → 342
    """
    normalised = normalise_german(corpus)
    words = pretokenise(normalised)
    
    word_freqs: Dict[str, int] = Counter(words)
    
    # Convert to character-level representation
    vocab: Dict[Tuple[str, ...], int] = {}
    for word, freq in word_freqs.items():
        char_seq = word_to_char_sequence(word)
        if char_seq:
            vocab[char_seq] = vocab.get(char_seq, 0) + freq
    
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Core BPE operations
# ─────────────────────────────────────────────────────────────────────────────

def get_pair_frequencies(
    vocab: Dict[Tuple[str, ...], int]
) -> Dict[Tuple[str, str], int]:
    """
    Count frequency of every adjacent pair across all words in vocab.
    
    This is the most computationally expensive step — O(V * L) where
    V = vocabulary size and L = average word length.
    
    For a 50MB German corpus this runs in a few seconds in pure Python.
    For production, this is vectorised with NumPy or Rust.
    """
    pairs: Dict[Tuple[str, str], int] = defaultdict(int)
    
    for word_chars, freq in vocab.items():
        # Slide a window of size 2 across the character sequence
        for i in range(len(word_chars) - 1):
            pair = (word_chars[i], word_chars[i + 1])
            pairs[pair] += freq
    
    return dict(pairs)


def merge_pair(
    vocab: Dict[Tuple[str, ...], int],
    pair: Tuple[str, str]
) -> Dict[Tuple[str, ...], int]:
    """
    Merge all occurrences of `pair` in vocab into a single token.
    
    Example: merge ('u', 'n') turns
    ('H', 'a', 'u', 'n', 'd</w>') → ('H', 'a', 'un', 'd</w>')
    
    This is applied to every word in the vocabulary simultaneously —
    we are updating the vocabulary representation, not the raw text.
    """
    new_vocab: Dict[Tuple[str, ...], int] = {}
    bigram = pair[0] + pair[1]          # the merged token string
    
    for word_chars, freq in vocab.items():
        new_word: List[str] = []
        i = 0
        while i < len(word_chars):
            # If current and next token match the pair, merge them
            if (i < len(word_chars) - 1 and
                    word_chars[i] == pair[0] and
                    word_chars[i + 1] == pair[1]):
                new_word.append(bigram)
                i += 2
            else:
                new_word.append(word_chars[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    
    return new_vocab


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: The BPE training loop
# ─────────────────────────────────────────────────────────────────────────────

class BPETokeniser:
    """
    Byte Pair Encoding tokeniser trained from scratch.
    
    Attributes:
        vocab_size:   target vocabulary size (character vocab + num_merges)
        merges:       ordered list of (pair, merged_token) — the merge rules
        token_to_id:  mapping from token string to integer id
        id_to_token:  reverse mapping
        special_tokens: <unk>, <pad>, <bos>, <eos>
    """
    
    SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[Tuple[str, str], str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._merge_dict: Dict[Tuple[str, str], str] = {}
        self.is_trained = False
    
    def train(self, corpus: str, verbose: bool = True) -> None:
        """
        Train BPE on a text corpus.
        
        The training process:
        1. Build character-level vocabulary from corpus
        2. Repeatedly find the most frequent adjacent pair
        3. Merge that pair into a new token
        4. Record the merge rule
        5. Stop when target vocab_size is reached
        
        Args:
            corpus:  raw text string (will be normalised internally)
            verbose: print progress every 500 merges
        """
        if verbose:
            print(f"Training BPE tokeniser | target vocab size: {self.vocab_size}")
        
        # Build initial character vocabulary
        vocab = build_vocab_from_corpus(corpus)
        
        # Collect all unique base characters
        base_chars: set = set()
        for word_chars in vocab.keys():
            base_chars.update(word_chars)
        
        # Start token set: special tokens + base characters
        all_tokens = self.SPECIAL_TOKENS + sorted(base_chars)
        num_merges_needed = self.vocab_size - len(all_tokens)
        
        if verbose:
            print(f"Base character vocabulary: {len(base_chars)} characters")
            print(f"Merges to perform: {num_merges_needed}")
        
        if num_merges_needed <= 0:
            raise ValueError(
                f"vocab_size={self.vocab_size} is too small. "
                f"Need at least {len(all_tokens)} for base characters."
            )
        
        # ── BPE merge loop ──────────────────────────────────────────
        for merge_idx in range(num_merges_needed):
            # Find the most frequent pair
            pairs = get_pair_frequencies(vocab)
            if not pairs:
                if verbose:
                    print(f"No more pairs to merge after {merge_idx} merges.")
                break
            
            best_pair = max(pairs, key=lambda p: pairs[p])
            best_freq = pairs[best_pair]
            merged_token = best_pair[0] + best_pair[1]
            
            # Apply the merge
            vocab = merge_pair(vocab, best_pair)
            self.merges.append((best_pair, merged_token))
            self._merge_dict[best_pair] = merged_token
            all_tokens.append(merged_token)
            
            if verbose and (merge_idx + 1) % 500 == 0:
                print(
                    f"  Merge {merge_idx+1:>5}/{num_merges_needed} | "
                    f"merged {best_pair[0]!r} + {best_pair[1]!r} → "
                    f"{merged_token!r} | freq: {best_freq}"
                )
        
        # ── Build token ↔ id mappings ───────────────────────────────
        self.token_to_id = {tok: i for i, tok in enumerate(all_tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        self.is_trained = True
        
        if verbose:
            print(f"\nTraining complete. Final vocabulary size: {len(self.token_to_id)}")
    
    # ── Encoding ────────────────────────────────────────────────────
    
    def _encode_word(self, word: str) -> List[str]:
        """
        Apply learned BPE merges to a single word.
        
        Start with character-level split, then apply merges in the
        same order they were learned during training.
        
        This is the crucial bit: we replay the merge history, not
        re-run the frequency counting. Encoding is deterministic and
        fast — O(merges * word_length).
        """
        chars = list(word_to_char_sequence(word))
        if not chars:
            return []
        
        # Apply merges greedily in training order
        for pair, merged in self.merges:
            i = 0
            new_chars: List[str] = []
            while i < len(chars):
                if (i < len(chars) - 1 and
                        chars[i] == pair[0] and
                        chars[i + 1] == pair[1]):
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        
        return chars
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode a text string to a list of token ids.
        
        Pipeline: normalise → pretokenise → BPE encode each word →
        convert tokens to ids → optionally wrap with <bos>/<eos>
        """
        if not self.is_trained:
            raise RuntimeError("Tokeniser not trained. Call .train() first.")
        
        normalised = normalise_german(text)
        words = pretokenise(normalised)
        
        unk_id = self.token_to_id['<unk>']
        token_ids: List[int] = []
        
        if add_special_tokens:
            token_ids.append(self.token_to_id['<bos>'])
        
        for word in words:
            subword_tokens = self._encode_word(word)
            for tok in subword_tokens:
                token_ids.append(
                    self.token_to_id.get(tok, unk_id)
                )
        
        if add_special_tokens:
            token_ids.append(self.token_to_id['<eos>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token ids back to a string.
        
        Remove </w> markers and strip special tokens.
        Reconstruct word boundaries from </w> markers.
        """
        special = set(self.SPECIAL_TOKENS)
        tokens = [
            self.id_to_token.get(i, '<unk>')
            for i in token_ids
            if self.id_to_token.get(i, '') not in special
        ]
        
        # Reconstruct text from subword tokens
        text = ''.join(tokens)
        text = text.replace('</w>', ' ').strip()
        return text
    
    # ── Metrics ─────────────────────────────────────────────────────
    
    def fertility(self, text: str) -> float:
        """
        Fertility = average number of tokens per whitespace-separated word.
        
        English GPT-4: ~1.3
        German GPT-4:  ~2.1
        This tokeniser on German with vocab 8k: expect ~1.8–2.4
        
        Lower is better — fewer tokens per word means:
        - Longer effective context window
        - Cheaper attention (O(n²) in sequence length)
        - More semantic content per token
        """
        words = text.split()
        if not words:
            return 0.0
        total_tokens = sum(
            len(self._encode_word(w)) for w in words
        )
        return total_tokens / len(words)
    
    def analyse_compound_word(self, word: str) -> None:
        """
        Show how a German compound word is segmented.
        """
        tokens = self._encode_word(word)
        ids = [self.token_to_id.get(t, self.token_to_id['<unk>']) for t in tokens]
        print(f"\nWord: {word}")
        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"  IDs: {ids}")
    
    # ── Persistence ─────────────────────────────────────────────────
    
    def save(self, path: str) -> None:
        """Save tokeniser state to JSON."""
        data = {
            'vocab_size': self.vocab_size,
            'merges': [
                [list(pair), merged]
                for pair, merged in self.merges
            ],
            'token_to_id': self.token_to_id,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokeniser saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BPETokeniser':
        """Load tokeniser from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(vocab_size=data['vocab_size'])
        tok.merges = [(tuple(pair), merged) for pair, merged in data['merges']]
        tok._merge_dict = {tuple(pair): merged for pair, merged in tok.merges}
        tok.token_to_id = data['token_to_id']
        tok.id_to_token = {int(i): t for t, i in tok.token_to_id.items()}
        tok.is_trained = True
        return tok