"""Adapter satisfying rag-local-eval-loop's embedder interface
(embed/embed_one/get_model) by wrapping speculative_rag.embedding.Embedder,
the same dense model (intfloat/multilingual-e5-small, see
speculative_rag.config.SETTINGS.embedding_model) used elsewhere in this
project for dense work. The production webapp's hot path actually retrieves
with BM25 (see speculative_rag.sparse_embedding / speculative_rag.pipeline's
own docstring) -- this dense model is what's available as a standalone
embed(texts)-style function, so it's what the eval loop's own throwaway
FAISS index gets built from. Query vs passage prefixing matches
Embedder.embed_query / Embedder.embed_passages exactly, since e5 models are
trained with those literal prefixes and score worse without them.
"""
import numpy as np

from speculative_rag.embedding import Embedder

_embedder: Embedder | None = None


def get_model() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    model = get_model()
    prefixed = [f"passage: {t}" for t in texts]
    return model.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)


def embed_one(text: str) -> np.ndarray:
    model = get_model()
    return model.model.encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False)
