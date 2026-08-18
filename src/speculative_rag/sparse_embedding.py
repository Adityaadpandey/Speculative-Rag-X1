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

from dataclasses import dataclass

from fastembed import SparseTextEmbedding

from speculative_rag.hindi_stemmer import stem_hindi_text

# fastembed.sparse.bm25.supported_languages only covers a fixed Snowball
# stemmer list, and Hindi isn't on it. disable_stemmer=True skips stemming
# AND stopword removal entirely regardless of the language string passed,
# so "english" here is just a harmless placeholder for languages with no
# stemmer.
_STEMMED_LANGUAGES = {"en": "english", "ta": "tamil"}


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

    def _preprocess(self, text: str) -> str:
        return stem_hindi_text(text) if self.language == "hi" else text

    def embed_passages(self, texts: list[str]) -> list[SparseVectorData]:
        texts = [self._preprocess(t) for t in texts]
        return [SparseVectorData(v.indices.tolist(), v.values.tolist()) for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> SparseVectorData:
        vector = next(iter(self._model.query_embed([self._preprocess(text)])))
        return SparseVectorData(vector.indices.tolist(), vector.values.tolist())
