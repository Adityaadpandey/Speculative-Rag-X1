"""Interactive extractive-answer CLI over an already-built collection.
Text query in -> extractive answer + guardrail decision + per-stage latency.
Read-only, never rebuilds the collection. Usage:
    uv run python -m scripts.answer [language] [strategy]
Defaults to the SETTINGS.default_language / default_strategy collection (hi_native).
"""

import sys

from speculative_rag.config import SETTINGS, get_qdrant_client
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.retrieval import Retriever
from speculative_rag.sparse_embedding import SparseEmbedder


def main() -> None:
    language = sys.argv[1] if len(sys.argv) > 1 else SETTINGS.default_language
    strategy = sys.argv[2] if len(sys.argv) > 2 else SETTINGS.default_strategy
    collection_name = f"{language}_{strategy}"

    print(f"Loading embedder and connecting to collection '{collection_name}'...")
    embedder = SparseEmbedder(language)
    client = get_qdrant_client()

    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        print(f"No such collection '{collection_name}'. Available: {available}")
        print("Run `uv run python -m scripts.run_recall_eval` first to build the index.")
        return

    retriever = Retriever(client, collection_name)

    print("Type a question (in the collection's language), Ctrl+D to quit.\n")
    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            print()
            break
        if not query:
            continue

        result = run_pipeline(
            query,
            embedder,
            retriever,
            off_topic_threshold=SETTINGS.off_topic_threshold,
            groundedness_threshold=SETTINGS.groundedness_threshold,
            top_k=SETTINGS.answer_top_k,
        )

        print(f"\noutcome: {result.outcome.value}")
        if result.outcome == RagOutcome.ANSWERED:
            print(f"answer: {result.answer.text}")
            print(
                f"source passage: {result.answer.source_passage_id}  "
                f"score={result.answer.retrieval_score:.3f}  "
                f"groundedness={result.answer.groundedness_score:.3f}"
            )
        else:
            print(f"refused: {result.refusal_reason}")

        print("latency (ms):")
        for timing in result.timings:
            print(f"  {timing.stage:24s} {timing.elapsed_ms:7.2f}")
        print(f"  {'TOTAL':24s} {result.total_ms:7.2f}\n")


if __name__ == "__main__":
    main()
