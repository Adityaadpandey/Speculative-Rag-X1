"""BM25 sparse query/passage encoding (fastembed's `Qdrant/bm25`: a
tokenizer + term counts, no neural weights -- around 100MB RSS measured,
vs. roughly 1GB for the prior dense e5-small pipeline) so the deployed
webapp fits on a free 512MB host. IDF weighting happens server-side
(Qdrant's sparse-vector `idf` modifier), so this only ever emits raw
term-frequency vectors.

Checked against the dense baseline on the existing recall eval before
switching over: ties or beats it for English (recall@1 0.327 vs 0.337,
recall@10 0.957 vs 0.867) but is a real regression for Hindi (recall@1
~0.14 vs 0.327) since fastembed ships no Hindi stemmer. A conservative
rule-based light stemmer (hindi_stemmer.py) runs ahead of the embedder for
Hindi text only, applied identically at index and query time, to recover
some of that gap without pulling in a model."""

import unicodedata
from dataclasses import dataclass

from fastembed import SparseTextEmbedding

from speculative_rag.hindi_stemmer import stem_hindi_text

# fastembed.sparse.bm25.supported_languages only covers a fixed Snowball
# stemmer list, and Hindi isn't on it. disable_stemmer=True skips stemming
# AND stopword removal entirely regardless of the language string passed,
# so "english" here is just a harmless placeholder for languages with no
# stemmer.
_STEMMED_LANGUAGES = {"en": "english", "ta": "tamil"}


def _indic_safe_words(text: str) -> list[str]:
    """fastembed's BM25 cleans text with re.sub(r"[^\w\s]", ...) and then
    tokenizes on r"[^\w]", and Python's \w doesn't count combining marks
    (Unicode category M) as word characters. Every Devanagari and Tamil
    vowel sign is one, so words got shredded into consonant fragments
    before BM25 ever saw them:
        "भारत की राजधानी"  ->  ['भ', 'रत', 'क', 'र', 'जध', 'न']
    Keeping letters, digits AND combining marks gives whole words back:
        "भारत की राजधानी"  ->  ['भारत', 'की', 'राजधानी']"""
    return "".join(
        c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in text.lower()
    ).split()


@dataclass
class SparseVectorData:
    indices: list[int]
    values: list[float]


class SparseEmbedder:
    def __init__(self, language: str):
        self.language = language
        stemmer_language = _STEMMED_LANGUAGES.get(language)
        self._model = SparseTextEmbedding(
            "Qdrant/bm25",
            language=stemmer_language or "english",
            disable_stemmer=stemmer_language is None,
        )
        # English keeps fastembed's stock text handling (so the existing
        # en_native index stays valid); every other language is tokenized by
        # _indic_safe_words and then handed back to fastembed's own stemming,
        # term-frequency and hashing, so the vectors keep the same format.
        # ponytail: _stem/_term_frequency are fastembed internals; the
        # tokenization tests fail loudly if an upgrade renames them.
        self._bm25 = None if language == "en" else self._model.model

    def _preprocess(self, text: str) -> str:
        return stem_hindi_text(text) if self.language == "hi" else text

    def _terms(self, text: str) -> list[str]:
        return self._bm25._stem(_indic_safe_words(self._preprocess(text)))

    def embed_passages(self, texts: list[str]) -> list[SparseVectorData]:
        if self._bm25 is not None:
            weights = [self._bm25._term_frequency(self._terms(t)) for t in texts]
            return [SparseVectorData(list(w), list(w.values())) for w in weights]
        texts = [self._preprocess(t) for t in texts]
        return [SparseVectorData(v.indices.tolist(), v.values.tolist()) for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> SparseVectorData:
        if self._bm25 is not None:
            ids = sorted({self._bm25.compute_token_id(t) for t in self._terms(text)})
            return SparseVectorData(ids, [1.0] * len(ids))
        vector = next(iter(self._model.query_embed([self._preprocess(text)])))
        return SparseVectorData(vector.indices.tolist(), vector.values.tolist())
