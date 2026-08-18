"""Async-only dense rerank over an already-retrieved BM25 candidate pool.

Not a retrieval stage -- doesn't touch Qdrant or the corpus index. Takes the
top-K sparse candidates the hot path already fetched and reorders them by
dense cosine similarity, so a query/passage pair that's semantically close
but lexically mismatched (BM25's weak spot) can still surface. Only ever
encodes the ~20 short candidate texts already in hand plus the query, not
the corpus -- so this stays cheap per-request even though the underlying
dense model itself is too heavy (~1GB RSS) for the free-tier hot-path
process. Import this lazily (see speculative_rag.embedding's own docstring):
loading it at module scope would pull sentence-transformers into every
process that imports this module, including ones that never call it.
"""

import numpy as np

from speculative_rag.embedding import Embedder
from speculative_rag.pipeline_schemas import ScoredChunk


def dense_rerank(query_text: str, candidates: list[ScoredChunk], embedder: Embedder) -> list[ScoredChunk]:
    """Return `candidates` sorted by dense cosine similarity to the query,
    descending. Ties never occur in practice (cosine similarity over a
    384-dim continuous embedding), so no explicit tiebreak is needed.
    Embeddings are already L2-normalized by Embedder, so a plain dot
    product is the cosine similarity."""
    if not candidates:
        return candidates

    query_vector = np.array(embedder.embed_query(query_text))
    passage_vectors = np.array(embedder.embed_passages([c.chunk.text for c in candidates]))
    similarities = passage_vectors @ query_vector

    order = np.argsort(-similarities)
    return [candidates[i] for i in order]
