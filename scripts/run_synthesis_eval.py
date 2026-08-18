"""Does async multi-passage LLM synthesis recover queries the hot path marks
INSUFFICIENT_EVIDENCE (no single retrieved sentence answers the query)?
Runs the real pipeline over a sample, and for every INSUFFICIENT_EVIDENCE
row, additionally calls speculative_rag.reflect.synthesize_answer_llm over
the full top-k candidate set to see whether the evidence collectively
answers the query even though no single passage does.

Makes one real Anthropic API call per INSUFFICIENT_EVIDENCE row, so this
runs on a small sample (SETTINGS.reflect_latency_sample_size, default 30).

Usage:
    uv run python -m scripts.run_synthesis_eval [language] [strategy]
Defaults to hi/native. Requires ANTHROPIC_API_KEY and an existing collection.
"""

import sys
import time

import pandas as pd

from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS, get_qdrant_client
from speculative_rag.data import load_validation_sample
from speculative_rag.latency import summarize
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.reflect import synthesize_answer_llm
from speculative_rag.retrieval import Retriever
from speculative_rag.sparse_embedding import SparseEmbedder
from speculative_rag.text_utils import tokenize

_GOLD_ANSWER_FIELD = {
    "Translated_passages": "Answer",
    "English_passages": "Eng_Answer",
}


def word_overlap_f1(predicted: str, gold: str) -> float:
    pred_tokens = tokenize(predicted)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def run_eval(language: str, strategy: str) -> None:
    source = next(s for s in LANGUAGE_SOURCES if s.language == language)
    gold_field = _GOLD_ANSWER_FIELD[source.text_field]
    collection_name = f"{language}_{strategy}"

    print(f"Loading embedder and connecting to collection '{collection_name}'...")
    embedder = SparseEmbedder(language)
    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        print(f"No such collection '{collection_name}'. Run scripts.run_recall_eval first.")
        return
    retriever = Retriever(client, collection_name)

    sample_size = SETTINGS.reflect_latency_sample_size
    df = load_validation_sample(source.parquet_path, sample_size, SETTINGS.sample_seed)

    rows = []
    synthesis_latencies_ms: list[float] = []
    insufficient_count = 0
    for i, (_, row) in enumerate(df.iterrows()):
        query_text = row[source.query_field]
        gold_answer = row[gold_field]

        result = run_pipeline(
            query_text,
            embedder,
            retriever,
            off_topic_threshold=SETTINGS.off_topic_threshold,
            groundedness_threshold=SETTINGS.groundedness_threshold,
            top_k=SETTINGS.answer_top_k,
            evidence_top_n=SETTINGS.evidence_top_n,
        )
        if result.outcome != RagOutcome.INSUFFICIENT_EVIDENCE:
            continue
        insufficient_count += 1

        query_vector = embedder.embed_query(query_text)
        candidates = retriever.search(query_vector, k=SETTINGS.answer_top_k).scored_chunks
        if not candidates:
            continue

        synth_start = time.perf_counter()
        verdict = synthesize_answer_llm(query_text, [c.chunk.text for c in candidates])
        synth_ms = (time.perf_counter() - synth_start) * 1000.0
        synthesis_latencies_ms.append(synth_ms)

        f1 = word_overlap_f1(verdict.answer_text, gold_answer) if verdict.answerable else 0.0
        rows.append(
            {
                "query_id": int(row["query_id"]),
                "query": query_text,
                "gold_answer": gold_answer,
                "recovered": verdict.answerable,
                "synthesized_answer": verdict.answer_text,
                "f1": f1,
            }
        )
        print(f"  [{i + 1}/{sample_size}] processed", end="\r")

    print()
    report_df = pd.DataFrame(rows)
    n = len(report_df)
    recovered_n = int(report_df["recovered"].sum()) if n else 0

    lines = [
        f"# Multi-passage synthesis recovery eval — {collection_name}, "
        f"{sample_size} sampled queries, {insufficient_count} were INSUFFICIENT_EVIDENCE\n"
    ]
    if n == 0:
        lines.append("\nNo INSUFFICIENT_EVIDENCE rows in this sample — nothing to recover.\n")
    else:
        lines.append("\n## Recovery\n")
        lines.append("| metric | value |\n|---|---|\n")
        lines.append(f"| INSUFFICIENT_EVIDENCE rows attempted | {n} |\n")
        lines.append(f"| recovered (synthesis found an answer) | {recovered_n} ({recovered_n / n * 100:.1f}%) |\n")
        lines.append(
            f"| mean word-overlap F1 vs gold answer (recovered rows) | "
            f"{report_df[report_df['recovered']]['f1'].mean():.3f} |\n"
            if recovered_n
            else "| mean word-overlap F1 vs gold answer (recovered rows) | n/a |\n"
        )

        latency_stats = summarize(synthesis_latencies_ms)
        lines.append("\n## Synthesis call latency (async, not counted against the 200ms RAG latency budget)\n")
        lines.append("| metric | value |\n|---|---|\n")
        lines.append(f"| P50 (ms) | {latency_stats['p50']:.1f} |\n")
        lines.append(f"| P70 (ms) | {latency_stats['p70']:.1f} |\n")
        lines.append(f"| P100 (ms) | {latency_stats['p100']:.1f} |\n")

    report_text = "".join(lines)
    output_path = SETTINGS.results_dir / f"synthesis_eval_{collection_name}.md"
    output_path.write_text(report_text)
    print(report_text)
    print(f"Saved to {output_path}")


def main() -> None:
    language = sys.argv[1] if len(sys.argv) > 1 else SETTINGS.default_language
    strategy = sys.argv[2] if len(sys.argv) > 2 else SETTINGS.default_strategy
    run_eval(language, strategy)


if __name__ == "__main__":
    main()
