"""Conservative rule-based light stemmer for Hindi.

fastembed's BM25 (Qdrant/bm25) has no Hindi entry in its stemmer table --
disable_stemmer=True for Hindi skips stemming entirely (see
sparse_embedding.py), so "लड़का" (boy) and "लड़के"/"लड़कों" (boy, inflected
forms) are completely different terms to BM25 with zero lexical overlap.
That's a real, measured hit to Hindi recall (see results/embedding_model_
comparison.md / chunking_comparison.md: Hindi BM25 recall@1 ~0.14 vs a
dense encoder's 0.327 on the same corpus/queries).

This strips the most common, unambiguous noun/verb inflectional suffixes
before text ever reaches the sparse embedder, applied identically at index
and query time so both sides land on the same stemmed term. Deliberately
small and conservative rather than a full suffix table (e.g. the
Ramanathan & Rao list): an incorrect stem that merges two unrelated words
is worse than under-stemming, since it silently corrupts BM25's term
statistics. Longest-suffix-first with a minimum remaining-stem length,
same shape as a standard Porter-style stemmer.
"""

import re

# (suffix, minimum stem length remaining after stripping it). Ordered
# longest-first so a word matching a longer suffix doesn't get truncated by
# a shorter one that happens to also match its tail (e.g. "ियों" should
# strip whole, not just its final "ों").
_SUFFIXES: tuple[tuple[str, int], ...] = (
    # Feminine oblique/plural: लड़कियों (girls, oblique) -> लड़क
    ("ियों", 2),
    # Common verb/participle endings: जाएंगे, करेंगे (future) -> stem
    ("ेंगे", 2), ("ेंगी", 2), ("ूंगा", 2), ("ूंगी", 2),
    # Oblique plural / verb agreement: लड़कों (boys, oblique), करते -> stem
    ("ों", 2), ("ाओं", 2), ("ाएं", 2),
    # Verb aspect/infinitive: करता/करती/करते (habitual), करना (infinitive),
    # करने (oblique infinitive), किया (perfective)
    ("ता", 2), ("ती", 2), ("ते", 2), ("ना", 2), ("ने", 2), ("या", 2),
    # Bare case/plural vowel markers: लड़का->लड़क, लड़के->लड़क, लड़की->लड़क
    ("ा", 2), ("े", 2), ("ी", 2), ("ों", 2), ("ें", 2),
)

_DEVANAGARI_WORD_RE = re.compile(r"[ऀ-ॿ]+")


def stem_word(word: str) -> str:
    """Strip the single longest matching suffix, if the remaining stem
    still meets that suffix's minimum length. Non-Devanagari input (digits,
    Latin text mixed into a Hindi query) passes through unchanged."""
    for suffix, min_stem_len in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem_len:
            return word[: -len(suffix)]
    return word


def stem_hindi_text(text: str) -> str:
    """Apply stem_word to every Devanagari word in `text`, leaving
    whitespace, punctuation, and non-Devanagari runs (numbers, Latin text)
    untouched. Must be applied identically at both index and query time --
    a stemmed passage matched against an unstemmed query defeats the point."""
    return _DEVANAGARI_WORD_RE.sub(lambda m: stem_word(m.group(0)), text)
