"""Phase 2 of the prefilter+rerank evaluation (see the prefilter section of
docs/superpowers/specs/2026-08-18-extraction-quality-and-prefilter-design.md):
paired LLM evaluation on the production recovery path, prefilter off vs. on.

Calls the ACTUAL production function, reflect.rerank_and_generate_llm --
not rerank_passages_llm plus a separate select_answer_span call, which is
what this script used to do. That measured a DIFFERENT function than the one
answer_orchestrator.resolve_recovery actually calls in production (which
selects AND generates in one round trip); every comparison the old version
produced was measuring the wrong thing. This version also runs the same
post-generation groundedness check resolve_recovery runs, so the reported
numbers reflect one full production recovery attempt, not just the
selection step.

Makes two real Anthropic API calls per query per variant (rerank+generate,
then a groundedness verification call), times two -- prefilter off and
prefilter on, same candidate pool per query -- so this runs on the smaller
sample (SETTINGS.reflect_latency_sample_size, default 30), not the 300-query
baseline. Run scripts.run_prefilter_recall_eval FIRST: it's the free go/no-go
gate for whether the prefilter is even safe to spend this budget evaluating.

Usage:
    uv run python -m scripts.run_llm_rerank_eval [language] [strategy]
Defaults to hi/native. Requires ANTHROPIC_API_KEY and an existing collection.
"""

import sys
import time

import anthropic
import pandas as pd

from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS, get_qdrant_client
from speculative_rag.data import load_validation_sample, passage_id_for
from speculative_rag.generation import prefilter_rerank_candidates
from speculative_rag.latency import summarize
from speculative_rag.pipeline_schemas import ScoredChunk
from speculative_rag.reflect import (
    _MODEL,
    _RERANK_AND_GENERATE_PROMPT_TEMPLATE,
    RerankCandidate,
    check_groundedness_llm,
    rerank_and_generate_llm,
)
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


def _count_prompt_tokens(query_text: str, candidates: list[RerankCandidate]) -> int:
    """Token count of the EXACT prompt reflect.rerank_and_generate_llm sends
    -- reuses its own template rather than approximating, since the point is
    to measure the real production prompt cost."""
    client = anthropic.Anthropic()
    candidates_block = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(candidates))
    response = client.messages.count_tokens(
        model=_MODEL,
        messages=[
            {
                "role": "user",
                "content": _RERANK_AND_GENERATE_PROMPT_TEMPLATE.format(
                    query_text=query_text, candidates_block=candidates_block
                ),
            }
        ],
    )
    return response.input_tokens


def _run_variant(
    query_text: str,
    candidates: list[ScoredChunk],
    relevant_ids: set[str],
    gold_answer: str,
) -> dict:
    """One rerank_and_generate_llm + check_groundedness_llm pass over a
    (possibly prefiltered) candidate pool -- the same two-call shape
    answer_orchestrator.resolve_recovery uses for one recovery attempt."""
    rerank_candidates = [RerankCandidate(source_id=c.chunk.chunk_id, text=c.chunk.text) for c in candidates]
    by_chunk_id = {c.chunk.chunk_id: c for c in candidates}

    start = time.perf_counter()
    verdict = rerank_and_generate_llm(query_text, rerank_candidates)

    selected_ids = {
        by_chunk_id[s.source_id].chunk.source_passage_id for s in verdict.selected_sources if s.source_id in by_chunk_id
    }
    llm_selected_gold_recall = bool(selected_ids & relevant_ids)

    grounded = None
    answers_query = None
    if verdict.answerable and verdict.answer_text:
        evidence_texts = [
            by_chunk_id[s.source_id].chunk.text for s in verdict.selected_sources if s.source_id in by_chunk_id
        ]
        groundedness_verdict = check_groundedness_llm(query_text, verdict.answer_text, " ".join(evidence_texts))
        grounded = groundedness_verdict.grounded
        answers_query = groundedness_verdict.answers_query
    recovery_latency_ms = (time.perf_counter() - start) * 1000.0

    f1 = word_overlap_f1(verdict.answer_text, gold_answer) if verdict.answerable and verdict.answer_text else 0.0

    return {
        "candidate_count": len(candidates),
        "llm_selected_gold_recall": llm_selected_gold_recall,
        "answer_f1": f1,
        "answerable": verdict.answerable,
        "grounded": grounded,
        "answers_query": answers_query,
        "prompt_tokens": _count_prompt_tokens(query_text, rerank_candidates),
        "recovery_latency_ms": recovery_latency_ms,
    }


def _summarize_variant(df: pd.DataFrame, label: str) -> list[str]:
    latency_stats = summarize(df["recovery_latency_ms"].tolist())
    answerable = df[df["answerable"]]
    grounded_rate = answerable["grounded"].mean() * 100 if len(answerable) else 0.0
    answers_query_rate = answerable["answers_query"].mean() * 100 if len(answerable) else 0.0
    return [
        f"| mean word-overlap F1 vs gold answer | {label} | {df['answer_f1'].mean():.3f} |\n",
        f"| LLM-selected gold-passage recall | {label} | {df['llm_selected_gold_recall'].mean() * 100:.1f}% |\n",
        f"| mean candidate count seen by the LLM | {label} | {df['candidate_count'].mean():.1f} |\n",
        f"| mean prompt tokens | {label} | {df['prompt_tokens'].mean():.0f} |\n",
        f"| answerable rate | {label} | {df['answerable'].mean() * 100:.1f}% |\n",
        f"| grounded rate (among answerable) | {label} | {grounded_rate:.1f}% |\n",
        f"| answers_query rate (among answerable) | {label} | {answers_query_rate:.1f}% |\n",
        f"| recovery latency P50 / P70 / P100 (ms) | {label} | "
        f"{latency_stats['p50']:.0f} / {latency_stats['p70']:.0f} / {latency_stats['p100']:.0f} |\n",
    ]


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

    off_rows = []
    on_rows = []
    pool_recall_hits = 0
    n = 0
    for i, (_, row) in enumerate(df.iterrows()):
        passages = row["passages"]
        texts = list(passages[source.text_field])
        selected = [int(x) for x in passages["is_selected"]]
        relevant_texts = {texts[j] for j, sel in enumerate(selected) if sel == 1}
        relevant_ids = {passage_id_for(language, t) for t in relevant_texts}

        query_text = row[source.query_field]
        gold_answer = row[gold_field]

        query_vector = embedder.embed_query(query_text)
        candidates = retriever.search(query_vector, k=SETTINGS.answer_top_k).scored_chunks
        if not candidates:
            continue
        n += 1

        pool_ids = {c.chunk.source_passage_id for c in candidates}
        if pool_ids & relevant_ids:
            pool_recall_hits += 1

        prefiltered = prefilter_rerank_candidates(
            query_text,
            candidates,
            lexical_top_n=SETTINGS.rerank_prefilter_lexical_top_n,
            retrieval_reserve=SETTINGS.rerank_prefilter_retrieval_reserve,
        )

        off_rows.append(_run_variant(query_text, candidates, relevant_ids, gold_answer))
        on_rows.append(_run_variant(query_text, prefiltered, relevant_ids, gold_answer))
        print(f"  [{i + 1}/{sample_size}] processed", end="\r")

    print()
    off_df = pd.DataFrame(off_rows)
    on_df = pd.DataFrame(on_rows)

    lines = [
        f"# LLM-rerank+regenerate eval (Phase 2, production path) — {collection_name}, {n} validation queries\n"
    ]
    lines.append(
        "\nCalls the actual production function (reflect.rerank_and_generate_llm) followed by the same "
        "groundedness verification answer_orchestrator.resolve_recovery runs. Paired comparison: prefilter "
        "off vs. on, same retrieved pool per query.\n"
    )
    lines.append(
        f"\nRetrieved gold-passage recall (pool-level, before any LLM step): {pool_recall_hits}/{n} "
        f"({(pool_recall_hits / n * 100) if n else 0.0:.1f}%)\n"
    )
    lines.append("\n## Metrics\n")
    lines.append("| metric | variant | value |\n|---|---|---|\n")
    lines.extend(_summarize_variant(off_df, "prefilter off"))
    lines.extend(_summarize_variant(on_df, "prefilter on"))

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
