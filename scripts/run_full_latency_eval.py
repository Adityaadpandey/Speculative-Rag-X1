"""Three-way latency report: STT-only, RAG-only (reused verbatim from
results/latency_rag.md, not re-measured here), and async-groundedness-only.
Each leg is an independent distribution -- there's no combined end-to-end
number, on purpose, since the three legs run at completely different points
in the request lifecycle. Costs real money per request (Sarvam + Anthropic
calls), so sample sizes are SETTINGS.stt_latency_sample_size /
reflect_latency_sample_size (30 each), far smaller than the RAG-only leg's
200.

Usage: uv run python -m scripts.run_full_latency_eval
"""

import time
from pathlib import Path

from qdrant_client import QdrantClient

from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS
from speculative_rag.data import build_query_records, load_validation_sample
from speculative_rag.embedding import Embedder
from speculative_rag.latency import summarize
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.reflect import check_groundedness_llm
from speculative_rag.retrieval import Retriever
from speculative_rag.stt import transcribe_audio

# Reused verbatim from results/latency_rag.md (200 validation queries,
# hi_native collection) -- not re-measured here.
RAG_ONLY_P50_MS = 7.16
RAG_ONLY_P70_MS = 7.45
RAG_ONLY_P100_MS = 43.17
RAG_ONLY_SAMPLE_SIZE = 200

STT_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_hindi.wav"
STT_FIXTURE_LANGUAGE_CODE = "hi-IN"


def _language_source(language: str):
    return next(source for source in LANGUAGE_SOURCES if source.language == language)


def measure_stt_only(sample_size: int) -> list[float]:
    latencies = []
    for _ in range(sample_size):
        start = time.perf_counter()
        transcribe_audio(STT_FIXTURE_PATH, STT_FIXTURE_LANGUAGE_CODE)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def measure_reflect_only(sample_size: int, embedder, retriever) -> list[float]:
    source = _language_source(SETTINGS.default_language)
    # Over-sample: not every validation query reaches ANSWERED, and only
    # ANSWERED results have a groundedness check to time.
    df = load_validation_sample(source.parquet_path, sample_size * 2, SETTINGS.sample_seed)
    records = build_query_records(df, source.language, source.text_field)

    latencies: list[float] = []
    for record in records:
        if len(latencies) >= sample_size:
            break
        result = run_pipeline(
            record.query_text,
            embedder,
            retriever,
            off_topic_threshold=SETTINGS.off_topic_threshold,
            groundedness_threshold=SETTINGS.groundedness_threshold,
            top_k=SETTINGS.answer_top_k,
        )
        if result.outcome != RagOutcome.ANSWERED:
            continue

        query_vector = embedder.embed_query(record.query_text)
        retrieval = retriever.search(query_vector, k=1)
        source_text = retrieval.top.chunk.text if retrieval.top is not None else ""

        start = time.perf_counter()
        check_groundedness_llm(result.answer.text, source_text)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def render_report(stt_latencies: list[float], reflect_latencies: list[float]) -> str:
    stt_stats = summarize(stt_latencies)
    reflect_stats = summarize(reflect_latencies)
    lines = [
        "# Three-way latency report -- STT / RAG / async-groundedness",
        "",
        "Each leg below is an independent distribution; there is no combined",
        "end-to-end latency number.",
        "",
        f"## STT-only ({len(stt_latencies)} calls, Sarvam speech-to-text)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| P50 (ms) | {stt_stats['p50']:.2f} |",
        f"| P70 (ms) | {stt_stats['p70']:.2f} |",
        f"| P100 (ms) | {stt_stats['p100']:.2f} |",
        "",
        f"## RAG-only (reused verbatim from results/latency_rag.md, {RAG_ONLY_SAMPLE_SIZE} validation queries -- not re-measured)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| P50 (ms) | {RAG_ONLY_P50_MS:.2f} |",
        f"| P70 (ms) | {RAG_ONLY_P70_MS:.2f} |",
        f"| P100 (ms) | {RAG_ONLY_P100_MS:.2f} |",
        "",
        f"## Async-groundedness-only ({len(reflect_latencies)} calls, Claude Sonnet 5, non-blocking)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| P50 (ms) | {reflect_stats['p50']:.2f} |",
        f"| P70 (ms) | {reflect_stats['p70']:.2f} |",
        f"| P100 (ms) | {reflect_stats['p100']:.2f} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print(f"Measuring STT-only latency ({SETTINGS.stt_latency_sample_size} calls against the committed fixture)...")
    stt_latencies = measure_stt_only(SETTINGS.stt_latency_sample_size)

    language = SETTINGS.default_language
    strategy = SETTINGS.default_strategy
    collection_name = f"{language}_{strategy}"
    embedder = Embedder()
    client = QdrantClient(path=str(SETTINGS.qdrant_path))
    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        print(f"No such collection '{collection_name}'. Available: {available}")
        print("Run `uv run python -m scripts.run_recall_eval` first to build the index.")
        return
    retriever = Retriever(client, collection_name)

    print(f"Measuring async-groundedness-only latency ({SETTINGS.reflect_latency_sample_size} calls)...")
    reflect_latencies = measure_reflect_only(SETTINGS.reflect_latency_sample_size, embedder, retriever)

    report = render_report(stt_latencies, reflect_latencies)
    SETTINGS.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = SETTINGS.results_dir / "latency_full.md"
    output_path.write_text(report)
    print(report)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
