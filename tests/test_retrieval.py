import pytest
from qdrant_client import QdrantClient

from speculative_rag.indexing import SparseChunkIndex
from speculative_rag.retrieval import Retriever
from speculative_rag.schemas import Chunk
from speculative_rag.sparse_embedding import SparseVectorData


def _chunk(pid, text):
    return Chunk(
        chunk_id=f"{pid}:native:0",
        text=text,
        language="en",
        query_type="DESCRIPTION",
        source_passage_id=pid,
        strategy="native",
    )


def _sparse(indices, values):
    return SparseVectorData(indices=indices, values=values)


@pytest.mark.slow
def test_retriever_returns_scored_chunks_ranked_by_similarity(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    # Seed a fixture collection with SparseChunkIndex (test-only helper).
    index = SparseChunkIndex(client, collection_name="en_native")
    index.add(
        [_chunk("p1", "near"), _chunk("p2", "far")],
        # p2 shares only token 2 with the query (partial overlap, still
        # returned by sparse search); p1 shares both, so it must outrank p2.
        [_sparse([1, 2], [1.0, 1.0]), _sparse([2, 3], [1.0, 1.0])],
    )

    retriever = Retriever(client, "en_native")
    result = retriever.search(_sparse([1, 2], [1.0, 1.0]), k=2)

    assert len(result.scored_chunks) == 2
    assert result.top.chunk.source_passage_id == "p1"
    assert result.top_score > result.scored_chunks[1].score


@pytest.mark.slow
def test_retriever_respects_k(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    index = SparseChunkIndex(client, collection_name="en_native")
    index.add(
        [_chunk(f"p{i}", f"text{i}") for i in range(5)],
        [_sparse([1, i + 10], [1.0, 1.0]) for i in range(5)],
    )

    retriever = Retriever(client, "en_native")
    result = retriever.search(_sparse([1], [1.0]), k=3)

    assert len(result.scored_chunks) == 3
