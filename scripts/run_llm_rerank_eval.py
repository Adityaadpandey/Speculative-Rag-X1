"""Does the async LLM rerank (outside the 200ms latency budget) actually fix
the rank-1 bottleneck that cheap lexical reranking made worse? Runs the same
end-to-end answer-quality measurement as scripts/run_answer_quality_eval.py,
but picks the answer source via speculative_rag.reflect.select_best_candidate_llm
over the top-k candidates instead of trusting embedding rank 1 or word-overlap.

Makes one real Anthropic API call per query, so this runs on a much smaller
sample than the 300-query baseline (SETTINGS.reflect_latency_sample_size,
default 30) and reports the rerank call's own latency percentiles
separately, never counted against the RAG-pipeline budget.

Usage:
    uv run python -m scripts.run_llm_rerank_eval [language] [strategy]
Defaults to hi/native. Requires ANTHROPIC_API_KEY and an existing collection.
"""

import sys

import pandas as pd

from qdrant_client import QdrantClient

from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS
from speculative_rag.data import load_validation_sample, passage_id_for
from speculative_rag.embedding import Embedder
from speculative_rag.generation import select_answer_span
from speculative_rag.latency import summarize
from speculative_rag.reflect import select_best_candidate_llm
from speculative_rag.retrieval import Retriever
from speculative_rag.text_utils import tokenize
import time

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
    embedder = Embedder()
    client = QdrantClient(path=str(SETTINGS.qdrant_path))
    if not client.collection_exists(collection_name):
        print(f"No such collection '{collection_name}'. Run scripts.run_recall_eval first.")
        return
    retriever = Retriever(client, collection_name)

    sample_size = SETTINGS.reflect_latency_sample_size
    df = load_validation_sample(source.parquet_path, sample_size, SETTINGS.sample_seed)

    rows = []
    rerank_latencies_ms: list[float] = []
    for i, (_, row) in enumerate(df.iterrows()):
        passages = row["passages"]
        texts = list(passages[source.text_field])
        selected = [int(x) for x in passages["is_selected"]]
        relevant_texts = {texts[j] for j, sel in enumerate(selected) if sel == 1}
        relevant_ids = {passage_id_for(language, t) for t in relevant_texts}

        query_text = row["query"] if source.text_field == "Translated_passages" else row["Eng_Query"]
        gold_answer = row[gold_field]

        query_vector = embedder.embed_query(query_text)
        retrieval = retriever.search(query_vector, k=SETTINGS.answer_top_k)
        candidates = retrieval.scored_chunks
        if not candidates:
            continue

        rerank_start = time.perf_counter()
        verdict = select_best_candidate_llm(query_text, [c.chunk.text for c in candidates])
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        rerank_latencies_ms.append(rerank_ms)

        chosen = candidates[verdict.best_index]
        answer_text = select_answer_span(query_text, chosen.chunk.text)
        f1 = word_overlap_f1(answer_text, gold_answer)
        sourced_from_gold_passage = chosen.chunk.source_passage_id in relevant_ids

        # baseline for the SAME query/candidate set, for a fair paired comparison
        baseline_chunk = candidates[0]
        baseline_answer = select_answer_span(query_text, baseline_chunk.chunk.text)
        baseline_f1 = word_overlap_f1(baseline_answer, gold_answer)
        baseline_sourced = baseline_chunk.chunk.source_passage_id in relevant_ids

        rows.append(
            {
                "query_id": int(row["query_id"]),
                "query": query_text,
                "gold_answer": gold_answer,
                "llm_answer": answer_text,
                "llm_f1": f1,
                "llm_sourced_from_gold_passage": sourced_from_gold_passage,
                "baseline_f1": baseline_f1,
                "baseline_sourced_from_gold_passage": baseline_sourced,
            }
        )
        print(f"  [{i + 1}/{sample_size}] processed", end="\r")

    print()
    report_df = pd.DataFrame(rows)
    n = len(report_df)

    latency_stats = summarize(rerank_latencies_ms)

    lines = [f"# LLM-rerank answer-quality eval — {collection_name}, {n} validation queries\n"]
    lines.append("Paired comparison against embedding-rank-1 baseline, same candidate set per query.\n")
    lines.append("\n## Answer quality\n")
    lines.append("| metric | baseline (rank-1) | LLM rerank |\n|---|---|---|\n")
    lines.append(
        f"| mean word-overlap F1 vs gold answer | {report_df['baseline_f1'].mean():.3f} | "
        f"{report_df['llm_f1'].mean():.3f} |\n"
    )
    lines.append(
        f"| % answer sourced from a gold-relevant passage | "
        f"{report_df['baseline_sourced_from_gold_passage'].mean() * 100:.1f}% | "
        f"{report_df['llm_sourced_from_gold_passage'].mean() * 100:.1f}% |\n"
    )

    lines.append("\n## Rerank call latency (async, not counted against the 200ms RAG latency budget)\n")
    lines.append("| metric | value |\n|---|---|\n")
    lines.append(f"| P50 (ms) | {latency_stats['p50']:.1f} |\n")
    lines.append(f"| P70 (ms) | {latency_stats['p70']:.1f} |\n")
    lines.append(f"| P100 (ms) | {latency_stats['p100']:.1f} |\n")

    flips_to_correct = report_df[
        (~report_df["baseline_sourced_from_gold_passage"]) & (report_df["llm_sourced_from_gold_passage"])
    ]
    flips_to_wrong = report_df[
        (report_df["baseline_sourced_from_gold_passage"]) & (~report_df["llm_sourced_from_gold_passage"])
    ]
    lines.append(
        f"\nrerank fixed {len(flips_to_correct)} queries that rank-1 got wrong, "
        f"and broke {len(flips_to_wrong)} queries that rank-1 got right.\n"
    )

    report_text = "".join(lines)
    output_path = SETTINGS.results_dir / f"llm_rerank_eval_{collection_name}.md"
    output_path.write_text(report_text)
    print(report_text)
    print(f"Saved to {output_path}")


def main() -> None:
    language = sys.argv[1] if len(sys.argv) > 1 else SETTINGS.default_language
    strategy = sys.argv[2] if len(sys.argv) > 2 else SETTINGS.default_strategy
    run_eval(language, strategy)


if __name__ == "__main__":
    main()
