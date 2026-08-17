import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, PointStruct, SparseVector, SparseVectorParams, VectorParams

from speculative_rag.schemas import Chunk
from speculative_rag.sparse_embedding import SparseVectorData

SPARSE_VECTOR_NAME = "bm25"


class SparseChunkIndex:
    """Sparse-vector counterpart of ChunkIndex, for BM25 collections. Kept
    separate rather than a mode flag on ChunkIndex, since dense collections
    still exist (run_recall_eval.py / run_embedding_model_eval.py's
    throwaway model-comparison collections) and the two vector shapes don't
    share a meaningful code path beyond "upsert points, then query"."""

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self._ensure_clean_collection()

    def _ensure_clean_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={},
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
        )

    def add(self, chunks: list[Chunk], vectors: list[SparseVectorData]) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector={SPARSE_VECTOR_NAME: SparseVector(indices=vector.indices, values=vector.values)},
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: SparseVectorData, k: int) -> list[Chunk]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(indices=query_vector.indices, values=query_vector.values),
            using=SPARSE_VECTOR_NAME,
            limit=k,
        ).points
        return [Chunk(**r.payload) for r in results]


class ChunkIndex:
    def __init__(self, client: QdrantClient, collection_name: str, vector_size: int):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_clean_collection()

    def _ensure_clean_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], k: int) -> list[Chunk]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
        ).points
        return [Chunk(**r.payload) for r in results]
