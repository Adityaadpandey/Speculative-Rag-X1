"""Read-only retrieval against an existing collection. Unlike
indexing.ChunkIndex (whose constructor deletes and recreates the
collection), Retriever only queries -- it never mutates the index."""

from qdrant_client import QdrantClient

from speculative_rag.pipeline_schemas import RetrievalResult, ScoredChunk
from speculative_rag.schemas import Chunk


class Retriever:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def search(self, query_vector: list[float], k: int) -> RetrievalResult:
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
        ).points
        scored = [ScoredChunk(chunk=Chunk(**point.payload), score=point.score) for point in points]
        return RetrievalResult(scored_chunks=scored)
