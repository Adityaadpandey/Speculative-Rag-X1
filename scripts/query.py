"""Interactive retrieval CLI over an already-built collection from
scripts/run_recall_eval.py. Retrieval only -- no answer generation, see
scripts/answer.py for that. Usage: uv run python -m scripts.query [language] [strategy]
"""

import sys

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from speculative_rag.config import get_qdrant_client
from speculative_rag.indexing import SPARSE_VECTOR_NAME
from speculative_rag.schemas import Chunk
from speculative_rag.sparse_embedding import SparseEmbedder, SparseVectorData


def search(client: QdrantClient, collection_name: str, query_vector: SparseVectorData, k: int = 5):
    results = client.query_points(
        collection_name=collection_name,
        query=SparseVector(indices=query_vector.indices, values=query_vector.values),
        using=SPARSE_VECTOR_NAME,
        limit=k,
    ).points
    return [(Chunk(**r.payload), r.score) for r in results]


def main() -> None:
    language = sys.argv[1] if len(sys.argv) > 1 else "hi"
    strategy = sys.argv[2] if len(sys.argv) > 2 else "native"
    collection_name = f"{language}_{strategy}"

    print(f"Loading embedder and connecting to collection '{collection_name}'...")
    embedder = SparseEmbedder(language)
    client = get_qdrant_client()

    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        print(f"No such collection '{collection_name}'. Available: {available}")
        print("Run `uv run python -m scripts.run_recall_eval` first to build the index.")
        return

    print("Type a question (in the collection's language), Ctrl+D to quit.\n")
    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            print()
            break
        if not query:
            continue
        query_vector = embedder.embed_query(query)
        for rank, (chunk, score) in enumerate(search(client, collection_name, query_vector), start=1):
            print(f"\n[{rank}] score={score:.3f}")
            print(chunk.text[:400])
        print()


if __name__ == "__main__":
    main()
