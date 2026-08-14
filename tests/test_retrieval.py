import pytest
from qdrant_client import QdrantClient

from speculative_rag.indexing import ChunkIndex
from speculative_rag.retrieval import Retriever
from speculative_rag.schemas import Chunk


def _chunk(pid, text):
    return Chunk(
        chunk_id=f"{pid}:native:0",
        text=text,
        language="en",
        query_type="DESCRIPTION",
        source_passage_id=pid,
        strategy="native",
    )


@pytest.mark.slow
def test_retriever_returns_scored_chunks_ranked_by_similarity(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    # Seed a fixture collection with Phase-1's ChunkIndex (test-only helper).
    index = ChunkIndex(client, collection_name="en_native", vector_size=3)
    index.add(
        [_chunk("p1", "near"), _chunk("p2", "far")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    retriever = Retriever(client, "en_native")
    result = retriever.search([0.9, 0.1, 0.0], k=2)

    assert len(result.scored_chunks) == 2
    assert result.top.chunk.source_passage_id == "p1"
    assert result.top_score > result.scored_chunks[1].score


@pytest.mark.slow
def test_retriever_respects_k(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    index = ChunkIndex(client, collection_name="en_native", vector_size=3)
    index.add(
        [_chunk(f"p{i}", f"text{i}") for i in range(5)],
        [[float(i), 0.0, 0.0] for i in range(5)],
    )

    retriever = Retriever(client, "en_native")
    result = retriever.search([2.0, 0.0, 0.0], k=3)

    assert len(result.scored_chunks) == 3
