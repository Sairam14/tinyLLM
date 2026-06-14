#!/usr/bin/env python3
"""Demo script for the BPE tokenizer."""

from bpe_tokenizer import BPETokeniser

# Sample German corpus
corpus = """
Das ist ein Test für den BPE-Tokenizer.
Deutsch ist eine Sprache mit Umlauten wie ä, ö, ü.
Die Bildung ist wichtig für die Zukunft.
Guten Morgen, wie geht es dir heute?
Das Donaudampfschifffahrtsgesellschaftskapitän ist ein langes deutsches Wort.
Ich liebe die deutsche Sprache und Kultur.
München, Berlin und Hamburg sind große Städte.
Die Schönheit der Natur ist beeindruckend.
"""

# Create and train tokenizer
tok = BPETokeniser(vocab_size=500)
tok.train(corpus, verbose=True)

# Test encoding
test_text = "Das ist ein Test."
print(f"\n{'='*60}")
print(f"Encoding: {test_text}")
encoded = tok.encode(test_text)
print(f"Token IDs: {encoded}")
print(f"Number of tokens: {len(encoded)}")

# Test decoding
decoded = tok.decode(encoded)
print(f"Decoded: {decoded}")

# Test fertility (tokens per word)
print(f"\n{'='*60}")
print(f"Fertility: {tok.fertility(test_text):.2f} tokens/word")

# Analyze compound word
print(f"\n{'='*60}")
tok.analyse_compound_word("Donaudampfschifffahrtsgesellschaftskapitän")

# Save tokenizer
print(f"\n{'='*60}")
tok.save("tokenizer.json")
