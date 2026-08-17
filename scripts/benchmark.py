"""Quick standalone latency smoke test for the hot-path pipeline (the
<200ms RAG-pipeline budget) -- fixed queries, no dataset/parquet dependency,
unlike scripts/run_latency_eval.py which samples the real validation set.
Use this for a fast local check after touching embed/retrieve/generate code;
use run_latency_eval.py for the real P50/P70/P100 report against gold data.

Usage:
    uv run python -m scripts.benchmark [n_queries] [language]

language defaults to "en" (collection en_native); pass "hi" to hit
hi_native instead. Both collections must already exist locally -- run
`uv run python -m scripts.run_recall_eval` first if they don't.
"""

import statistics
import sys
from collections import defaultdict

from speculative_rag.config import SETTINGS, get_qdrant_client
from speculative_rag.latency import percentile
from speculative_rag.pipeline import run_pipeline
from speculative_rag.retrieval import Retriever
from speculative_rag.sparse_embedding import SparseEmbedder

LATENCY_BUDGET_MS = 200.0

QUERIES = {
    "en": [
        "What is retrieval augmented generation?",
        "How does BM25 scoring work?",
        "What causes climate change?",
        "Who invented the telephone?",
        "How do vaccines work?",
        "What is the capital of France?",
        "Why is the sky blue?",
        "How does a search engine rank results?",
    ],
    "hi": [
        "भारत की राजधानी क्या है?",
        "प्रकाश संश्लेषण क्या है?",
        "टेलीफोन का आविष्कार किसने किया?",
        "जलवायु परिवर्तन का कारण क्या है?",
        "वैक्सीन कैसे काम करती है?",
        "आकाश नीला क्यों दिखता है?",
        "कंप्यूटर क्या है?",
        "सूर्य कितना बड़ा है?",
    ],
}


def warmup(embedder, retriever) -> None:
    run_pipeline(
        QUERIES["en"][0] if embedder.language == "en" else QUERIES["hi"][0],
        embedder,
        retriever,
        off_topic_threshold=SETTINGS.off_topic_threshold,
        groundedness_threshold=SETTINGS.groundedness_threshold,
        top_k=SETTINGS.answer_top_k,
    )


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    language = sys.argv[2] if len(sys.argv) > 2 else "en"
    collection_name = f"{language}_native"
    queries = QUERIES.get(language)
    if queries is None:
        print(f"No fixed queries for language '{language}'. Choices: {list(QUERIES)}")
        sys.exit(1)

    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        print(f"No such collection '{collection_name}'. Available: {available}")
        print("Run `uv run python -m scripts.run_recall_eval` first to build the index.")
        sys.exit(1)

    embedder = SparseEmbedder(language)
    retriever = Retriever(client, collection_name)

    print("Warming up (model load + first query)...")
    warmup(embedder, retriever)

    total_ms: list[float] = []
    stage_ms: dict[str, list[float]] = defaultdict(list)
    for i in range(n):
        query = queries[i % len(queries)]
        result = run_pipeline(
            query,
            embedder,
            retriever,
            off_topic_threshold=SETTINGS.off_topic_threshold,
            groundedness_threshold=SETTINGS.groundedness_threshold,
            top_k=SETTINGS.answer_top_k,
        )
        total_ms.append(result.total_ms)
        for timing in result.timings:
            stage_ms[timing.stage].append(timing.elapsed_ms)

    print(f"\nRan {n} queries against '{collection_name}'\n")
    print(f"{'stage':<24}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    for name, values in [*stage_ms.items(), ("total", total_ms)]:
        print(
            f"{name:<24}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 95):>8.2f}"
            f"{percentile(values, 99):>8.2f}"
        )

    p95_total = percentile(total_ms, 95)
    print(f"\nLatency budget: {LATENCY_BUDGET_MS}ms | p95 total: {p95_total:.2f}ms")
    if p95_total <= LATENCY_BUDGET_MS:
        print("PASS: within budget")
    else:
        print("FAIL: over budget")
        sys.exit(1)


if __name__ == "__main__":
    main()
