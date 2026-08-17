"""Read-only retrieval against an existing collection. Unlike
indexing.ChunkIndex (whose constructor deletes and recreates the
collection), Retriever only queries -- it never mutates the index."""

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from speculative_rag.indexing import SPARSE_VECTOR_NAME
from speculative_rag.pipeline_schemas import RetrievalResult, ScoredChunk
from speculative_rag.schemas import Chunk
from speculative_rag.sparse_embedding import SparseVectorData


class Retriever:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def search(self, query_vector: SparseVectorData, k: int) -> RetrievalResult:
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(indices=query_vector.indices, values=query_vector.values),
            using=SPARSE_VECTOR_NAME,
            limit=k,
        ).points
        scored = [ScoredChunk(chunk=Chunk(**point.payload), score=point.score) for point in points]
        return RetrievalResult(scored_chunks=scored)
