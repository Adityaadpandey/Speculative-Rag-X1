import pytest
from qdrant_client import QdrantClient

from speculative_rag.indexing import ChunkIndex, SparseChunkIndex
from speculative_rag.schemas import Chunk
from speculative_rag.sparse_embedding import SparseVectorData


def _chunk(cid, text, source_passage_id):
    return Chunk(
        chunk_id=cid,
        text=text,
        language="en",
        query_type="DESCRIPTION",
        source_passage_id=source_passage_id,
        strategy="native",
    )


@pytest.mark.slow
def test_add_and_search_returns_nearest_chunk(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    index = ChunkIndex(client, collection_name="test", vector_size=3)

    chunks = [
        _chunk("c1", "near", "p1"),
        _chunk("c2", "far", "p2"),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    index.add(chunks, vectors)

    results = index.search(query_vector=[0.9, 0.1, 0.0], k=1)

    assert len(results) == 1
    assert results[0].source_passage_id == "p1"


@pytest.mark.slow
def test_search_respects_k(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    index = ChunkIndex(client, collection_name="test", vector_size=3)

    chunks = [_chunk(f"c{i}", f"text{i}", f"p{i}") for i in range(5)]
    vectors = [[float(i), 0.0, 0.0] for i in range(5)]
    index.add(chunks, vectors)

    results = index.search(query_vector=[2.0, 0.0, 0.0], k=3)

    assert len(results) == 3


@pytest.mark.slow
def test_sparse_chunk_index_recreates_collection_on_reinit(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    index = SparseChunkIndex(client, collection_name="test_sparse")
    index.add([_chunk("c1", "near", "p1")], [SparseVectorData(indices=[1, 2], values=[1.0, 1.0])])
    assert client.get_collection("test_sparse").points_count == 1

    # Constructing again deletes and recreates -- same guarantee as ChunkIndex.
    SparseChunkIndex(client, collection_name="test_sparse")
    assert client.get_collection("test_sparse").points_count == 0
