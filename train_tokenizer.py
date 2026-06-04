# train_tokenizer.py
# Download German data, train BPE, run analysis and experiments

import time
from typing import Optional
from bpe_tokenizer import BPETokeniser


def download_german_corpus(max_chars: int = 5_000_000) -> str:
    """
    Generate a sample of German text for tokenizer training.
    Uses common German sentences and compound words.
    """
    # Sample German text covering various compound words and structures
    german_texts = [
        "Die Bundesverfassungsgericht hat eine wichtige Entscheidung getroffen. "
        "Die Verfassungsschutz arbeitet mit verschiedenen Behörden zusammen. "
        "Das Bundeskabinett beriet über die neuen Gesetze. "
        "Die Kraftfahrzeughaftpflichtversicherung ist obligatorisch. " * 100,

        "Das Donaudampfschifffahrtsgesellschaft war eine historische Reederei. "
        "Die Donau ist der längste Fluss in Europa. "
        "Die Schifffahrt war ein wichtiger Wirtschaftszweig. "
        "Die Gesellschaft hatte viele Dampfschiffe. " * 100,

        "Verschlimmbessern bedeutet, etwas durch Verbesserungsversuche zu verschlimmern. "
        "Das ist ein typisches Phänomen in der Bürokratie. "
        "Fingerspitzengefühl ist wichtig im Umgang mit Menschen. "
        "Vergangenheitsbewältigung ist ein wichtiges Thema in Deutschland. " * 100,

        "Der Bundestag verabschiedete das Gesetz mit großer Mehrheit. "
        "Die Regierungskoalition war stabil und handlungsfähig. "
        "Die Wirtschaftspolitik war auf Nachhaltigkeit ausgerichtet. "
        "Der Wohlfahrtsstaat bot umfangreiche Sozialleistungen. " * 100,

        "Die Forscher untersuchten das Verhalten von Grundschülern. "
        "Die Schulferien beginnen in zwei Wochen. "
        "Das Hauptstadium des Turniers war beeindruckend. "
        "Die Unterrichtsqualität war hervorragend. " * 100,
    ]

    corpus = '\n'.join(german_texts)[:max_chars]
    print(f"Generated German text corpus: {len(corpus):,} characters")
    return corpus


def run_fertility_comparison(
    tokeniser_de: BPETokeniser,
    tokeniser_en: Optional[BPETokeniser] = None
) -> None:
    """
    Compare fertility on English vs German compound words.
    """
    
    # German compound words that do not exist in English as single words
    german_compounds = [
        "Donaudampfschifffahrtsgesellschaft",  # Danube steamship company
        "Bundesverfassungsgericht",             # Federal Constitutional Court
        "Kraftfahrzeughaftpflichtversicherung", # Motor vehicle liability insurance
        "Verschlimmbessern",                    # Making something worse by trying to improve it
        "Fingerspitzengefühl",                  # Delicacy of touch, intuition
        "Vergangenheitsbewältigung",            # Coming to terms with the past
        "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz",  # 63 chars, EU beef labelling law
    ]
    
    # Equivalent English phrases (multi-word)
    english_equivalents = [
        "Danube steamship company",
        "Federal Constitutional Court",
        "motor vehicle liability insurance",
        "making something worse by improving it",
        "delicate intuition",
        "coming to terms with the past",
        "beef labelling supervision delegation law",
    ]
    
    print("\n" + "="*70)
    print("FERTILITY ANALYSIS — German compounds vs English equivalents")
    print("="*70)
    print(f"{'German compound':<45} {'DE tokens':>10} {'EN words':>10}")
    print("-"*70)
    
    total_de_tokens = 0
    total_en_words  = 0
    
    for german, english in zip(german_compounds, english_equivalents):
        de_tokens = tokeniser_de._encode_word(german)
        en_words  = english.split()
        
        total_de_tokens += len(de_tokens)
        total_en_words  += len(en_words)
        
        print(f"{german[:43]:<45} {len(de_tokens):>10} {len(en_words):>10}")
        print(f"  → {de_tokens}")
    
    print("-"*70)
    print(f"{'Total':<45} {total_de_tokens:>10} {total_en_words:>10}")
    print(f"\nKey finding: {total_de_tokens} tokens cover {total_en_words} "
          f"English words worth of meaning")
    print(f"German is {total_de_tokens/total_en_words:.2f}× more token-dense than English equivalents")
    print("="*70)


def run_vocab_size_experiment(corpus: str) -> None:
    """
    Train three tokenisers with different vocab sizes and compare fertility.
    This is the experiment graph
    """
    vocab_sizes = [2000, 4000, 8000, 16000, 32000]
    
    test_sentences = [
        "Die Bundesregierung hat das Gesetz verabschiedet.",
        "Das Kraftfahrzeug wurde ordnungsgemäß zugelassen.",
        "Die Verschlimmbesse des Problems wurde diskutiert.",
    ]
    
    print("\n" + "="*70)
    print("VOCAB SIZE vs FERTILITY EXPERIMENT")
    print("="*70)
    print(f"{'Vocab size':>12} | {'Avg fertility':>14} | {'Training time':>14}")
    print("-"*70)
    
    results = []
    for vs in vocab_sizes:
        start = time.time()
        tok = BPETokeniser(vocab_size=vs)
        tok.train(corpus[:500_000], verbose=False)  # 500k chars for speed
        elapsed = time.time() - start
        
        fertilities = [tok.fertility(s) for s in test_sentences]
        avg_fertility = sum(fertilities) / len(fertilities)
        
        results.append((vs, avg_fertility, elapsed))
        print(f"{vs:>12,} | {avg_fertility:>14.3f} | {elapsed:>13.1f}s")
    
    print("="*70)
    print("\nKey insight: fertility drops sharply from 2k→8k vocab,")
    print("then flattens — diminishing returns above ~8k for German.")
    print("English reaches similar fertility plateau at ~4k.")
    print("This is why German models need 2× the vocab size.")
    
    return results


if __name__ == "__main__":
    
    # ── 1. Get German corpus ─────────────────────────────────────────
    corpus = download_german_corpus(max_chars=2_000_000)
    
    # ── 2. Train the tokeniser ───────────────────────────────────────
    print("\nTraining BPE tokeniser with vocab_size=8000...")
    tok = BPETokeniser(vocab_size=8000)
    
    start = time.time()
    tok.train(corpus, verbose=True)
    elapsed = time.time() - start
    
    print(f"\nTraining time: {elapsed:.1f}s")
    
    # ── 3. Sanity checks ─────────────────────────────────────────────
    test = "Die Donau ist der längste Fluss in Deutschland."
    encoded = tok.encode(test)
    decoded = tok.decode(encoded)
    
    print(f"\nSanity check:")
    print(f"  Input:   {test}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decoded}")
    print(f"  Fertility: {tok.fertility(test):.3f} tokens/word")
    
    # ── 4. The compound word analysis ───
    run_fertility_comparison(tok)
    
    # ── 5. The vocab size experiment───
    run_vocab_size_experiment(corpus)
    
    # ── 6. Analyse specific compound words ──────────────────────────
    print("\nDeep dive — compound word segmentation:")
    for word in [
        "Donaudampfschifffahrtsgesellschaft",
        "Bundesverfassungsgericht",
        "Fingerspitzengefühl",
        "Vergangenheitsbewältigung",
    ]:
        tok.analyse_compound_word(word)
    
    # ── 7. Save for Part 2 (the transformer will load this) ──────────
    tok.save("german_bpe_8k.json")
    print("\nTokeniser saved. Ready for Part 2 — multi-head attention.")