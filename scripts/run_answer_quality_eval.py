"""End-to-end answer-quality eval: runs the real hot-path pipeline
(embed -> retrieve -> guardrails -> extractive answer) over a validation
sample and scores what actually comes back against the dataset's gold
`Answer`/`Eng_Answer` field.

This is deliberately different from scripts/run_recall_eval.py, which only
asks "did the gold passage show up in the top-k retrieved chunks?". Recall@k
being high (0.85-0.94 for hi/en) says retrieval finds the right passage; it
says nothing about whether the single sentence the extractive step picks out
of that passage is actually a good answer. This script measures that second,
separate question end to end, against already-built collections (read-only,
no re-indexing).

Usage:
    uv run python -m scripts.run_answer_quality_eval [language] [strategy]
Defaults to hi/native. Requires the collection to already exist
(`uv run python -m scripts.run_recall_eval` builds it).
"""

import os
import sys
import pandas as pd
from qdrant_client import QdrantClient

from speculative_rag.answer_status import combine_verdict
from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS
from speculative_rag.data import load_validation_sample
from speculative_rag.embedding import Embedder
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.reflect import check_groundedness_llm
from speculative_rag.retrieval import Retriever
from speculative_rag.text_utils import tokenize

_GOLD_ANSWER_FIELD = {
    "Translated_passages": "Answer",
    "English_passages": "Eng_Answer",
}


def word_overlap_f1(predicted: str, gold: str) -> float:
    """Approximate word-overlap F1 (token sets, not multisets) between the
    extractive answer and the dataset's gold answer sentence. Not an exact
    SQuAD-style metric (no repeated-token weighting), but it's cheap and
    language-agnostic since it uses the same Unicode tokenizer the
    pipeline's own groundedness check uses."""
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

    df = load_validation_sample(source.parquet_path, SETTINGS.sample_size, SETTINGS.sample_seed)

    rows = []
    outcome_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        passages = row["passages"]
        texts = list(passages[source.text_field])
        selected = [int(x) for x in passages["is_selected"]]
        relevant_texts = {texts[i] for i, sel in enumerate(selected) if sel == 1}

        query_text = row["query"] if source.text_field == "Translated_passages" else row["Eng_Query"]
        gold_answer = row[gold_field]

        result = run_pipeline(
            query_text,
            embedder,
            retriever,
            off_topic_threshold=SETTINGS.off_topic_threshold,
            groundedness_threshold=SETTINGS.groundedness_threshold,
            top_k=SETTINGS.answer_top_k,
        )

        outcome_counts[result.outcome.value] = outcome_counts.get(result.outcome.value, 0) + 1

        f1 = None
        sourced_from_gold_passage = None
        answer_text = None
        if result.outcome == RagOutcome.ANSWERED:
            answer_text = result.answer.text
            f1 = word_overlap_f1(answer_text, gold_answer)
            # Was the sentence extracted from a passage the dataset actually
            # marked relevant, or from a plausible-looking distractor?
            source_id = result.answer.source_passage_id
            from speculative_rag.data import passage_id_for

            relevant_ids = {passage_id_for(language, t) for t in relevant_texts}
            sourced_from_gold_passage = source_id in relevant_ids

        rows.append(
            {
                "query_id": int(row["query_id"]),
                "query_type": row["query_type"],
                "query": query_text,
                "gold_answer": gold_answer,
                "outcome": result.outcome.value,
                "answer": answer_text,
                "f1": f1,
                "sourced_from_gold_passage": sourced_from_gold_passage,
            }
        )

    report_df = pd.DataFrame(rows)
    answered = report_df[report_df["outcome"] == "answered"]

    verify_n = min(SETTINGS.reflect_latency_sample_size, len(answered))
    status_counts: dict[str, int] = {}
    if verify_n > 0 and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping async LLM verification.")
    elif verify_n > 0:
        print(f"Running async LLM verification on {verify_n} ANSWERED rows...")
        verify_sample = answered.sample(n=verify_n, random_state=SETTINGS.sample_seed)
        for _, r in verify_sample.iterrows():
            query_vector = embedder.embed_query(r["query"])
            top = retriever.search(query_vector, k=1).top
            source_text = top.chunk.text if top is not None else ""
            verdict = check_groundedness_llm(r["query"], r["answer"], source_text)
            status, _ = combine_verdict(RagOutcome.ANSWERED, verdict)
            status_counts[status.value] = status_counts.get(status.value, 0) + 1

    lines = [f"# Answer-quality eval — {collection_name}, {len(report_df)} validation queries\n"]
    lines.append("## Outcome breakdown\n")
    lines.append("| outcome | count |\n|---|---|\n")
    for outcome, count in sorted(outcome_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {outcome} | {count} |\n")

    if len(answered) > 0:
        mean_f1 = answered["f1"].mean()
        median_f1 = answered["f1"].median()
        pct_sourced = answered["sourced_from_gold_passage"].mean() * 100
        pct_f1_zero = (answered["f1"] == 0.0).mean() * 100
        lines.append("\n## Answer quality (among ANSWERED queries)\n")
        lines.append("| metric | value |\n|---|---|\n")
        lines.append(f"| mean word-overlap F1 vs gold answer | {mean_f1:.3f} |\n")
        lines.append(f"| median word-overlap F1 vs gold answer | {median_f1:.3f} |\n")
        lines.append(f"| % with zero word overlap vs gold answer | {pct_f1_zero:.1f}% |\n")
        lines.append(f"| % answer sourced from a gold-relevant passage | {pct_sourced:.1f}% |\n")

        if status_counts:
            lines.append(
                f"\n## Real answer status among ANSWERED rows "
                f"({verify_n}-row LLM-verified sample — this is what 'answered' actually means)\n"
            )
            lines.append(
                "The pipeline's ANSWERED outcome only means a candidate extractive answer passed cheap "
                "heuristic gates — it is not a correctness verdict. This table applies the async four-"
                "dimension LLM judge (faithfulness/answerability/completeness/relevance) to a sample of "
                "those rows and reports what they actually resolve to.\n"
            )
            lines.append("| final status | count | % of sample |\n|---|---|---|\n")
            for status, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {status} | {count} | {count / verify_n * 100:.1f}% |\n")

        worst = answered.sort_values("f1").head(10)
        lines.append("\n## 10 lowest-F1 examples (for manual inspection)\n")
        for _, r in worst.iterrows():
            lines.append(f"\n**query** ({r['query_type']}): {r['query']}\n")
            lines.append(f"- gold answer: {r['gold_answer']}\n")
            lines.append(f"- pipeline answer: {r['answer']}\n")
            lines.append(f"- f1={r['f1']:.3f}  sourced_from_gold_passage={r['sourced_from_gold_passage']}\n")

    report_text = "".join(lines)
    output_path = SETTINGS.results_dir / f"answer_quality_{collection_name}.md"
    output_path.write_text(report_text)
    print(report_text)
    print(f"Saved to {output_path}")


def main() -> None:
    language = sys.argv[1] if len(sys.argv) > 1 else SETTINGS.default_language
    strategy = sys.argv[2] if len(sys.argv) > 2 else SETTINGS.default_strategy
    run_eval(language, strategy)


if __name__ == "__main__":
    main()
