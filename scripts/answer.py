"""Interactive answer CLI over an already-built collection.

Default mode is the fast extractive path. Add ``--smart`` to run the existing
LLM validation/rerank/regenerate/synthesis recovery chain after retrieval, so
the final answer is selected for the question rather than simply copied from
the highest lexical match.

Usage:
    uv run python -m scripts.answer [language] [strategy] [--smart]
"""

import os
import sys

from speculative_rag.answer_orchestrator import resolve_recovery
from speculative_rag.config import SETTINGS, get_qdrant_client
from speculative_rag.pipeline import run_pipeline
from speculative_rag.retrieval import Retriever
from speculative_rag.sparse_embedding import SparseEmbedder


def main() -> None:
    smart = "--smart" in sys.argv[1:]
    positional = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    language = positional[0] if positional else SETTINGS.default_language
    strategy = positional[1] if len(positional) > 1 else SETTINGS.default_strategy
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

    if smart and not os.environ.get("ANTHROPIC_API_KEY"):
        print("--smart requested, but ANTHROPIC_API_KEY is not configured; using fast mode.")
        smart = False
    print(
        "Type a question (in the collection's language), Ctrl+D to quit. "
        + ("Smart recovery is enabled.\n" if smart else "\n")
    )
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
            evidence_top_n=SETTINGS.evidence_top_n,
        )

        print(f"\nfast outcome: {result.outcome.value}")
        if result.answer is not None:
            print(f"fast answer: {result.answer.text}")
        else:
            print(f"fast refusal: {result.refusal_reason}")

        if smart:
            try:
                resolved = resolve_recovery(
                    query,
                    result,
                    prefilter_enabled=SETTINGS.rerank_prefilter_enabled,
                    prefilter_lexical_top_n=SETTINGS.rerank_prefilter_lexical_top_n,
                    prefilter_retrieval_reserve=SETTINGS.rerank_prefilter_retrieval_reserve,
                )
                print(f"final status: {resolved.final_status.value}")
                if resolved.answer_text:
                    print(f"final answer ({resolved.answer_engine}): {resolved.answer_text}")
                else:
                    print(f"final refusal: {resolved.reason}")
            except Exception as exc:
                print(f"smart recovery failed; keeping fast answer: {exc}")

        print("latency (ms):")
        for timing in result.timings:
            print(f"  {timing.stage:24s} {timing.elapsed_ms:7.2f}")
        print(f"  {'TOTAL':24s} {result.total_ms:7.2f}\n")


if __name__ == "__main__":
    main()
