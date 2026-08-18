"""One-off calibration for scoring._BIGRAM_BONUS (see the shared scorer
section of docs/superpowers/specs/2026-08-18-extraction-quality-and-prefilter-design.md):
sweeps a small set of candidate bonus values against the validation
sample's answer-quality F1 (the same lexical metric run_answer_quality_eval.py
uses) and reports the winner. No LLM calls -- each query's top passage and
gold answer are fetched from Qdrant ONCE and cached, since the bonus value
only affects in-memory scoring, not retrieval, so sweeping several values
doesn't mean re-hitting Qdrant per value.

Usage:
    uv run python -m scripts.calibrate_bigram_bonus [language] [strategy]
Defaults to en/native. Requires an existing collection (see
scripts.run_recall_eval).
"""

import sys

import speculative_rag.scoring as scoring
from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS, get_qdrant_client
from speculative_rag.data import load_validation_sample
from speculative_rag.generation import select_answer_span
from speculative_rag.retrieval import Retriever
from speculative_rag.sparse_embedding import SparseEmbedder
from speculative_rag.text_utils import tokenize

_CANDIDATE_VALUES = [0.0, 0.2, 0.35, 0.5, 0.75, 1.0]

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


def run_calibration(language: str, strategy: str) -> None:
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

    df = load_validation_sample(source.parquet_path, SETTINGS.sample_size, SETTINGS.sample_seed)

    cases: list[tuple[str, str, str]] = []  # (query_text, top_passage_text, gold_answer)
    for i, (_, row) in enumerate(df.iterrows()):
        query_text = row[source.query_field]
        gold_answer = row[gold_field]
        query_vector = embedder.embed_query(query_text)
        top = retriever.search(query_vector, k=1).top
        if top is not None:
            cases.append((query_text, top.chunk.text, gold_answer))
        print(f"  [{i + 1}/{len(df)}] retrieved", end="\r")
    print()

    results = []
    for bonus in _CANDIDATE_VALUES:
        scoring._BIGRAM_BONUS = bonus
        f1s = []
        for query_text, passage_text, gold_answer in cases:
            answer = select_answer_span(query_text, passage_text, language)
            f1s.append(word_overlap_f1(answer, gold_answer) if answer is not None else 0.0)
        mean_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        results.append((bonus, mean_f1))
        print(f"bigram_bonus={bonus:<5} mean_f1={mean_f1:.4f}  (n={len(f1s)})")

    best_bonus, best_f1 = max(results, key=lambda r: r[1])
    print(f"\nWinner: bigram_bonus={best_bonus} (mean_f1={best_f1:.4f})")
    print("Update scoring._BIGRAM_BONUS to this value.")


def main() -> None:
    language = sys.argv[1] if len(sys.argv) > 1 else "en"
    strategy = sys.argv[2] if len(sys.argv) > 2 else SETTINGS.default_strategy
    run_calibration(language, strategy)


if __name__ == "__main__":
    main()
