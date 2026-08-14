"""RAG-pipeline-only latency analytics -- not the STT or async-check numbers,
which get their own report in run_full_latency_eval.py.

Samples validation queries in the default collection's language, runs the
hot-path pipeline per query against the already-built collection (read-only),
and reports P50/P70/P100 of total RAG-pipeline latency plus a
guardrail-outcome breakdown and the top-1 score distribution (to justify the
off-topic threshold). Shows whether the <200ms target holds.

Usage: uv run python -m scripts.run_latency_eval
"""

from qdrant_client import QdrantClient

from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS
from speculative_rag.data import build_query_records, load_validation_sample
from speculative_rag.embedding import Embedder
from speculative_rag.latency import summarize
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.retrieval import Retriever


def _language_source(language: str):
    return next(source for source in LANGUAGE_SOURCES if source.language == language)


def _run(text: str, embedder, retriever):
    return run_pipeline(
        text,
        embedder,
        retriever,
        off_topic_threshold=SETTINGS.off_topic_threshold,
        groundedness_threshold=SETTINGS.groundedness_threshold,
        top_k=SETTINGS.answer_top_k,
    )


def render_report(collection_name: str, latencies: list[float], outcomes: dict[str, int], top_scores: list[float]) -> str:
    stats = summarize(latencies)
    lines = [
        f"# RAG-pipeline-only latency — `{collection_name}`, {len(latencies)} validation queries",
        "",
        "| metric | value |",
        "|---|---|",
        f"| P50 (ms) | {stats['p50']:.2f} |",
        f"| P70 (ms) | {stats['p70']:.2f} |",
        f"| P100 (ms) | {stats['p100']:.2f} |",
        "| target | <200ms |",
        f"| P100 within target | {'YES' if stats['p100'] < 200 else 'NO'} |",
        "",
        "## Guardrail outcome breakdown",
        "",
        "| outcome | count |",
        "|---|---|",
        *[f"| {name} | {count} |" for name, count in outcomes.items()],
        "",
    ]
    if top_scores:
        ordered = sorted(top_scores)
        median = ordered[len(ordered) // 2]
        lines.append(
            f"top-1 retrieval score across scored queries: min={ordered[0]:.3f} "
            f"median={median:.3f} max={ordered[-1]:.3f} "
            f"(off_topic_threshold={SETTINGS.off_topic_threshold})"
        )
    else:
        lines.append("no queries reached the retrieval stage (all refused pre-embedding)")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    language = SETTINGS.default_language
    strategy = SETTINGS.default_strategy
    collection_name = f"{language}_{strategy}"

    source = _language_source(language)
    df = load_validation_sample(source.parquet_path, SETTINGS.latency_sample_size, SETTINGS.sample_seed)
    records = build_query_records(df, source.language, source.text_field)

    embedder = Embedder()
    client = QdrantClient(path=str(SETTINGS.qdrant_path))
    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        print(f"No such collection '{collection_name}'. Available: {available}")
        print("Run `uv run python -m scripts.run_recall_eval` first to build the index.")
        return
    retriever = Retriever(client, collection_name)

    # Warm the embedder and the on-disk index so the first timed query isn't
    # penalised by lazy model / segment loading.
    _run(records[0].query_text, embedder, retriever)

    latencies: list[float] = []
    top_scores: list[float] = []
    outcomes: dict[str, int] = {outcome.value: 0 for outcome in RagOutcome}
    for record in records:
        result = _run(record.query_text, embedder, retriever)
        latencies.append(result.total_ms)
        outcomes[result.outcome.value] += 1
        if result.top_score is not None:
            top_scores.append(result.top_score)

    report = render_report(collection_name, latencies, outcomes, top_scores)
    SETTINGS.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = SETTINGS.results_dir / "latency_rag.md"
    output_path.write_text(report)
    print(report)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
