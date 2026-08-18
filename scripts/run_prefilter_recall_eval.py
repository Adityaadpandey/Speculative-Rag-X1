"""Phase 1 of the prefilter evaluation (see the prefilter section of
docs/superpowers/specs/2026-08-18-extraction-quality-and-prefilter-design.md):
offline, no LLM calls. Runs generation.prefilter_rerank_candidates alone against the full
300-query validation sample and checks whether it EVER excludes a
gold-relevant passage that was actually present in the retrieved pool.

This is the actual go/no-go gate for flipping
config.SETTINGS.rerank_prefilter_enabled to True -- if this reports any
exclusions, the prefilter design needs revision before spending any LLM
budget on the paired evaluation in scripts/run_llm_rerank_eval.py. Free to
run (BM25 retrieval only, no Anthropic API calls).

Usage:
    uv run python -m scripts.run_prefilter_recall_eval [language] [strategy]
Defaults to hi/native. Requires an existing collection (see
scripts.run_recall_eval).
"""

import sys

import pandas as pd

from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS, get_qdrant_client
from speculative_rag.data import load_validation_sample, passage_id_for
from speculative_rag.generation import prefilter_rerank_candidates
from speculative_rag.retrieval import Retriever
from speculative_rag.sparse_embedding import SparseEmbedder


def run_eval(language: str, strategy: str) -> None:
    source = next(s for s in LANGUAGE_SOURCES if s.language == language)
    collection_name = f"{language}_{strategy}"

    print(f"Loading embedder and connecting to collection '{collection_name}'...")
    embedder = SparseEmbedder(language)
    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        print(f"No such collection '{collection_name}'. Run scripts.run_recall_eval first.")
        return
    retriever = Retriever(client, collection_name)

    df = load_validation_sample(source.parquet_path, SETTINGS.sample_size, SETTINGS.sample_seed)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        passages = row["passages"]
        texts = list(passages[source.text_field])
        selected = [int(x) for x in passages["is_selected"]]
        relevant_texts = {texts[j] for j, sel in enumerate(selected) if sel == 1}
        relevant_ids = {passage_id_for(language, t) for t in relevant_texts}

        query_text = row[source.query_field]
        query_vector = embedder.embed_query(query_text)
        candidates = retriever.search(query_vector, k=SETTINGS.answer_top_k).scored_chunks

        candidate_ids = {c.chunk.source_passage_id for c in candidates}
        gold_in_pool = bool(candidate_ids & relevant_ids)

        survivors = prefilter_rerank_candidates(
            query_text,
            candidates,
            lexical_top_n=SETTINGS.rerank_prefilter_lexical_top_n,
            retrieval_reserve=SETTINGS.rerank_prefilter_retrieval_reserve,
        )
        survivor_ids = {c.chunk.source_passage_id for c in survivors}
        gold_survived = bool(survivor_ids & relevant_ids) if gold_in_pool else None

        rows.append(
            {
                "query_id": int(row["query_id"]),
                "gold_in_pool": gold_in_pool,
                "gold_survived_prefilter": gold_survived,
                "pool_size": len(candidates),
                "survivor_count": len(survivors),
            }
        )
        print(f"  [{i + 1}/{len(df)}] processed", end="\r")
    print()

    report_df = pd.DataFrame(rows)
    with_gold = report_df[report_df["gold_in_pool"]]
    excluded = with_gold[with_gold["gold_survived_prefilter"] == False]  # noqa: E712

    lines = [f"# Prefilter recall eval (Phase 1, offline, no LLM calls) — {collection_name}, {len(report_df)} queries\n"]
    lines.append(
        "\nGo/no-go gate for config.SETTINGS.rerank_prefilter_enabled: this must show ZERO "
        "gold-passage exclusions before the flag can flip to True.\n"
    )
    lines.append("\n## Results\n")
    lines.append("| metric | value |\n|---|---|\n")
    lines.append(f"| queries with gold passage in the retrieved pool | {len(with_gold)} / {len(report_df)} |\n")
    lines.append(f"| gold-passage exclusions caused by the prefilter | {len(excluded)} |\n")
    lines.append(
        f"| mean pool size before / after prefilter | "
        f"{report_df['pool_size'].mean():.1f} / {report_df['survivor_count'].mean():.1f} |\n"
    )

    if len(excluded) == 0:
        lines.append("\n**PASS** — the prefilter never excluded a gold-relevant passage on this sample.\n")
    else:
        lines.append(
            f"\n**FAIL** — the prefilter excluded the gold-relevant passage on {len(excluded)} "
            "quer" + ("y" if len(excluded) == 1 else "ies") + ". Do not enable rerank_prefilter_enabled "
            "until this is resolved. Excluded query_ids: "
            + ", ".join(str(q) for q in excluded["query_id"].tolist())
            + "\n"
        )

    report_text = "".join(lines)
    output_path = SETTINGS.results_dir / f"prefilter_recall_eval_{collection_name}.md"
    output_path.write_text(report_text)
    print(report_text)
    print(f"Saved to {output_path}")


def main() -> None:
    language = sys.argv[1] if len(sys.argv) > 1 else SETTINGS.default_language
    strategy = sys.argv[2] if len(sys.argv) > 2 else SETTINGS.default_strategy
    run_eval(language, strategy)


if __name__ == "__main__":
    main()
