"""Deterministic RAG quality, retrieval, and latency benchmark.

This evaluates the real ``run_pipeline()`` output without an LLM judge. It
supports the actual MSMARCO-XI schema: ``Answer``/``Eng_Answer`` contain the
gold answer, while ``passages.is_selected`` labels gold passages.

Usage:
    uv run python -m scripts.testing path/to/validation.parquet 100 hi

For ``en``/``hi``/``ur``, the ``dataset`` argument is accepted for
backward-compatible invocation but its rows are not used. Those languages
are evaluated against the exact same seeded train+validation sample that
``ingest_production_corpus.py`` indexed into ``{language}_native`` --
otherwise most gold passages referenced by an arbitrary validation slice
were never indexed at all (only ~4.6% of the combined corpus is sampled
into the production collections), which craters retrieval and answer
metrics for reasons that have nothing to do with retrieval or generation
quality.

The per-query JSONL report includes the retrieved passage text, source IDs,
and scores so retrieval failures can be inspected directly.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from speculative_rag.config import SETTINGS, get_qdrant_client
from speculative_rag.data import _has_gold_passage, passage_id_for
from speculative_rag.latency import percentile
from speculative_rag.pipeline import run_pipeline
from speculative_rag.retrieval import Retriever
from speculative_rag.sparse_embedding import SparseEmbedder

from scripts.ingest_production_corpus import (
    SAMPLE_QUERIES,
    _TARGET_CORPUS,
    load_combined_sample,
)


LATENCY_BUDGET_MS = 200.0

PRODUCTION_LANGUAGES = set(_TARGET_CORPUS)

# Matches SETTINGS.answer_top_k so this benchmark measures the retrieval
# pool width production actually uses by default.
DEFAULT_TOP_K = SETTINGS.answer_top_k

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# -----------------------------
# Text / answer metrics
# -----------------------------

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokens(text: Any) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def token_prf(prediction: str, gold: str) -> tuple[float, float, float]:
    pred = tokens(prediction)
    ref = tokens(gold)

    if not pred and not ref:
        return 1.0, 1.0, 1.0
    if not pred or not ref:
        return 0.0, 0.0, 0.0

    pred_counts = Counter(pred)
    ref_counts = Counter(ref)

    overlap = sum((pred_counts & ref_counts).values())

    precision = overlap / len(pred)
    recall = overlap / len(ref)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    # O(len(a) * len(b)), fine for short QA answers.
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, gold: str) -> float:
    pred = tokens(prediction)
    ref = tokens(gold)

    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    lcs = lcs_length(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)

    if precision + recall == 0:
        return 0.0

    # Classic ROUGE-L F-measure with beta=1.
    return 2 * precision * recall / (precision + recall)


def best_against_gold(prediction: str, gold_answers: Sequence[str]) -> dict[str, float | str]:
    golds = [g for g in gold_answers if normalize_text(g)]
    if not golds:
        return {
            "gold": "",
            "exact_match": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "rouge_l": 0.0,
        }

    best: dict[str, float | str] | None = None

    for gold in golds:
        p, r, f1 = token_prf(prediction, gold)
        candidate = {
            "gold": gold,
            "exact_match": exact_match(prediction, gold),
            "precision": p,
            "recall": r,
            "f1": f1,
            "rouge_l": rouge_l(prediction, gold),
        }

        if best is None:
            best = candidate
            continue

        # Rank by F1 first, then ROUGE-L, then EM.
        score = (
            float(candidate["f1"]),
            float(candidate["rouge_l"]),
            float(candidate["exact_match"]),
        )
        best_score = (
            float(best["f1"]),
            float(best["rouge_l"]),
            float(best["exact_match"]),
        )
        if score > best_score:
            best = candidate

    assert best is not None
    return best


# -----------------------------
# Dataset loading
# -----------------------------

def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit(
                "Parquet support requires pyarrow. Install it with `uv add pyarrow`."
            ) from exc

        table = pq.read_table(path)
        return table.to_pylist()

    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON on line {line_no}: {exc}") from exc
        return rows

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Convenient support for {"data": [...]}.
            if isinstance(data.get("data"), list):
                return data["data"]
            raise SystemExit("JSON file must contain a list of rows or {'data': [...]}.")

    raise SystemExit(
        f"Unsupported dataset format: {path.suffix}. Use .parquet, .jsonl, or .json."
    )


def load_production_matched_rows(language: str, n: int) -> list[dict[str, Any]]:
    """Replay the exact seeded sample ingest_production_corpus.py indexed
    into ``{language}_native``, filtered to rows with a labeled gold
    passage. Evaluating against this instead of arbitrary validation rows
    guarantees every gold passage referenced by the benchmark actually
    exists in the collection being queried."""
    import pyarrow as pa

    corpus_language = _TARGET_CORPUS[language]
    combined = load_combined_sample(SAMPLE_QUERIES, SETTINGS.sample_seed, corpus_language)
    combined = combined[combined["passages"].apply(_has_gold_passage)]
    # DataFrame.to_dict("records") leaves nested struct/list columns (like
    # "passages") as numpy arrays instead of plain lists, which breaks the
    # `... or []` truthiness checks the row helpers below rely on. Route
    # through pyarrow, matching what load_rows() already does for parquet.
    return pa.Table.from_pandas(combined.head(n)).to_pylist()


def get_answers(row: dict[str, Any], language: str) -> list[str]:
    """Read the gold answer from the MSMARCO-XI language-specific field.

    The dataset does not have an ``answers`` column. It stores the native
    answer in ``Answer`` and the English answer in ``Eng_Answer``. Keeping a
    fallback makes the benchmark useful with the older MS MARCO-shaped JSON
    fixtures as well.
    """
    field = "Eng_Answer" if language == "en" else "Answer"
    answers = row.get(field)
    if answers is None:
        answers = row.get("answers", [])
    if answers is None:
        return []
    if isinstance(answers, str):
        return [answers]
    return [str(x) for x in answers if x is not None]


def get_gold_passages(
    row: dict[str, Any],
    language: str,
) -> list[str]:
    passages = row.get("passages", {})
    if not isinstance(passages, dict):
        return []

    selected = passages.get("is_selected", []) or []

    # MSMARCO-XI stores the original English and translated/native passage
    # text separately. Evaluate against the collection's language.
    if language == "en":
        texts = passages.get("English_passages", []) or []
    else:
        texts = passages.get("Translated_passages", []) or []

    out = []
    for text, flag in zip(texts, selected):
        try:
            chosen = int(flag) == 1
        except (TypeError, ValueError):
            chosen = bool(flag)

        if chosen and text is not None:
            out.append(str(text))

    return out


def get_gold_passage_indices(row: dict[str, Any]) -> list[int]:
    passages = row.get("passages", {})
    if not isinstance(passages, dict):
        return []

    selected = passages.get("is_selected", []) or []
    indices = []

    for i, flag in enumerate(selected):
        try:
            chosen = int(flag) == 1
        except (TypeError, ValueError):
            chosen = bool(flag)

        if chosen:
            indices.append(i)

    return indices


def get_gold_passage_ids(row: dict[str, Any], language: str) -> list[str]:
    """Return the same stable source IDs used when passages are indexed."""
    passages = row.get("passages", {})
    if not isinstance(passages, dict):
        return []

    text_field = "English_passages" if language == "en" else "Translated_passages"
    texts = passages.get(text_field, []) or []
    selected = passages.get("is_selected", []) or []
    ids = []
    for text, flag in zip(texts, selected):
        try:
            chosen = int(flag) == 1
        except (TypeError, ValueError):
            chosen = bool(flag)
        if chosen and text is not None:
            ids.append(passage_id_for(language, str(text)))
    return ids


# -----------------------------
# Pipeline result extraction
# -----------------------------

def first_attr(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value

    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]

    return None


def extract_answer(result) -> str:
    answer = getattr(result, "answer", None)

    if answer is None:
        return ""

    text = getattr(answer, "text", None)

    if text is not None:
        return str(text)

    return str(answer)


def normalize_passage_for_match(text: str) -> str:
    # Keep this conservative. Retrieval passages should normally match exactly
    # after whitespace normalization.
    return re.sub(r"\s+", " ", str(text).strip().lower())


def flatten_passage_candidates(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        for key in (
            "passage_text",
            "text",
            "content",
            "document",
            "chunk",
        ):
            if key in value and value[key] is not None:
                return [str(value[key])]

        # Some retrievers return {"documents": [...]} / {"contexts": [...]}.
        for key in ("documents", "contexts", "passages", "results", "hits"):
            if key in value:
                return flatten_passage_candidates(value[key])

        return []

    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(flatten_passage_candidates(item))
        return out

    return []


def extract_retrieved_chunks(result: Any) -> list[Any]:
    return list(getattr(result, "retrieved_chunks", []) or [])


def get_retrieved_passages(result: Any) -> list[str]:
    chunks = extract_retrieved_chunks(result)

    passages = []
    for scored in chunks:
        chunk = getattr(scored, "chunk", None)
        text = getattr(chunk, "text", None)

        if text is not None:
            passages.append(str(text))

    return passages


def get_retrieved_source_ids(result: Any) -> list[str]:
    chunks = extract_retrieved_chunks(result)

    source_ids = []
    for scored in chunks:
        chunk = getattr(scored, "chunk", None)
        source_id = getattr(chunk, "source_passage_id", None)

        if source_id is not None:
            source_ids.append(str(source_id))

    return source_ids


def get_retrieved_scores(result: Any) -> list[float]:
    chunks = extract_retrieved_chunks(result)

    scores = []
    for scored in chunks:
        score = safe_float(getattr(scored, "score", None))
        if score is not None:
            scores.append(score)

    return scores


def retrieval_rank_metrics(
    retrieved_passages: Sequence[str],
    gold_passages: Sequence[str],
    retrieved_source_ids: Sequence[str] = (),
    gold_source_ids: Sequence[str] = (),
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute retrieval metrics using indexed source IDs when available.

    Matching whole text is wrong for chunked indexes: a retrieved chunk is
    only part of the labeled passage. Source IDs are stable across chunking,
    so use them for the real benchmark and retain text matching as a fallback
    for legacy result objects.
    """
    base = {f"hit@{k}": 0.0 for k in ks}
    base.update({
        "mrr": 0.0,
        "retrieval_available": float(bool(retrieved_passages)),
        "retrieval_evaluable": float(bool(gold_passages or gold_source_ids)),
    })

    if not (gold_passages or gold_source_ids) or not retrieved_passages:
        return base

    if gold_source_ids and len(retrieved_source_ids) == len(retrieved_passages):
        gold_keys = {str(source_id) for source_id in gold_source_ids}
        retrieved_keys = [str(source_id) for source_id in retrieved_source_ids]
    else:
        gold_keys = {
            normalize_passage_for_match(p)
            for p in gold_passages
            if normalize_passage_for_match(p)
        }
        retrieved_keys = [normalize_passage_for_match(p) for p in retrieved_passages]

    first_rank = next(
        (idx for idx, key in enumerate(retrieved_keys, start=1) if key in gold_keys),
        None,
    )
    base.update({
        f"hit@{k}": float(first_rank is not None and first_rank <= k)
        for k in ks
    })
    base["mrr"] = 1.0 / first_rank if first_rank is not None else 0.0
    return base


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# -----------------------------
# Main benchmark
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("n_queries", type=int, nargs="?", default=100)
    parser.add_argument("language", nargs="?", default="en")
    parser.add_argument("--output", type=Path, default=Path("benchmark_quality.jsonl"))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    collection_name = f"{args.language}_native"

    if args.language in PRODUCTION_LANGUAGES:
        print(
            f"Ignoring '{args.dataset}': replaying the same seeded corpus "
            f"sample ingest_production_corpus.py indexed into "
            f"'{collection_name}' so every gold passage is retrievable.\n"
        )
        rows = load_production_matched_rows(args.language, args.n_queries)
    else:
        rows = load_rows(args.dataset)

    if not rows:
        raise SystemExit("Dataset is empty.")

    n = min(args.n_queries, len(rows))
    rows = rows[:n]

    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        raise SystemExit(
            f"No such collection '{collection_name}'. Available: {available}\n"
            "Build the collection first."
        )

    embedder = SparseEmbedder(args.language)
    retriever = Retriever(client, collection_name)
    query_field = "Eng_Query" if args.language == "en" else "query"

    print(f"Benchmarking {n} queries against '{collection_name}'")
    print("No LLM evaluator is used. Answer scoring is deterministic.\n")

    # Warm up model / DB connection.
    warmup_query = str(rows[0].get(query_field, ""))
    run_pipeline(
        warmup_query,
        embedder,
        retriever,
        off_topic_threshold=SETTINGS.off_topic_threshold,
        groundedness_threshold=SETTINGS.groundedness_threshold,
        top_k=args.top_k,
        evidence_top_n=SETTINGS.evidence_top_n,
    )

    totals: list[float] = []
    answer_em: list[float] = []
    answer_f1: list[float] = []
    answer_rouge: list[float] = []
    answer_precision: list[float] = []
    answer_recall: list[float] = []

    retrieval_metrics: dict[str, list[float]] = defaultdict(list)
    stage_ms: dict[str, list[float]] = defaultdict(list)

    rows_out: list[dict[str, Any]] = []
    retrieval_available_count = 0
    retrieval_evaluable_count = 0

    for i, row in enumerate(rows, start=1):
        query = str(row.get(query_field, "")).strip()
        if not query:
            continue

        gold_answers = get_answers(row, args.language)
        gold_passages = get_gold_passages(row, args.language)
        gold_passage_ids = get_gold_passage_ids(row, args.language)

        result = run_pipeline(
            query,
            embedder,
            retriever,
            off_topic_threshold=SETTINGS.off_topic_threshold,
            groundedness_threshold=SETTINGS.groundedness_threshold,
            top_k=args.top_k,
            evidence_top_n=SETTINGS.evidence_top_n,
        )

        prediction = extract_answer(result)
        answer_score = best_against_gold(prediction, gold_answers)

        totals.append(float(result.total_ms))
        answer_em.append(float(answer_score["exact_match"]))
        answer_f1.append(float(answer_score["f1"]))
        answer_rouge.append(float(answer_score["rouge_l"]))
        answer_precision.append(float(answer_score["precision"]))
        answer_recall.append(float(answer_score["recall"]))

        for timing in result.timings:
            stage_ms[timing.stage].append(float(timing.elapsed_ms))

        retrieved_passages = get_retrieved_passages(result)
        retrieved_source_ids = get_retrieved_source_ids(result)
        rmetrics = retrieval_rank_metrics(
            retrieved_passages,
            gold_passages,
            retrieved_source_ids=retrieved_source_ids,
            gold_source_ids=gold_passage_ids,
        )

        if rmetrics["retrieval_available"]:
            retrieval_available_count += 1
        if rmetrics["retrieval_evaluable"]:
            retrieval_evaluable_count += 1
            for name, value in rmetrics.items():
                if name not in {"retrieval_available", "retrieval_evaluable"}:
                    retrieval_metrics[name].append(value)

        row_out = {
            "index": i,
            "query_id": row.get("query_id"),
            "query": query,
            "prediction": prediction,
            "gold_answers": gold_answers,
            "best_gold": answer_score["gold"],
            "outcome": getattr(getattr(result, "outcome", None), "value", None),
            "refusal_reason": getattr(result, "refusal_reason", None),
            "top_score": safe_float(getattr(result, "top_score", None)),
            "answer_exact_match": answer_score["exact_match"],
            "answer_precision": answer_score["precision"],
            "answer_recall": answer_score["recall"],
            "answer_f1": answer_score["f1"],
            "answer_rouge_l": answer_score["rouge_l"],
            "gold_selected_passage_count": len(gold_passages),
            "gold_selected_passage_indices": get_gold_passage_indices(row),
            "gold_source_ids": gold_passage_ids,
            "retrieved_passage_count": len(retrieved_passages),
            "num_evidence_candidates": getattr(result, "num_evidence_candidates", 0),
            "retrieved_source_ids": retrieved_source_ids,
            "retrieved_scores": get_retrieved_scores(result),
            "retrieved_top_passages": retrieved_passages[:10],
            "retrieval": rmetrics,
            "total_ms": float(result.total_ms),
        }

        status = first_attr(result, ("status", "answer_status", "final_status"))
        if status is not None:
            row_out["status"] = str(status)

        for attr_name in (
            "grounded",
            "groundedness",
            "answerable",
            "fallback",
            "fallback_triggered",
        ):
            value = first_attr(result, (attr_name,))
            if value is not None:
                row_out[attr_name] = value

        rows_out.append(row_out)

        if i % 10 == 0 or i == n:
            print(
                f"[{i:>4}/{n}] "
                f"F1={answer_f1[-1]:.3f} "
                f"EM={answer_em[-1]:.3f} "
                f"latency={totals[-1]:.1f}ms"
            )

    if not totals:
        raise SystemExit("No valid queries were evaluated.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== ANSWER QUALITY ===")
    print(f"Queries:              {len(totals)}")
    print(f"Exact Match:          {statistics.mean(answer_em):.4f}")
    print(f"Token Precision:      {statistics.mean(answer_precision):.4f}")
    print(f"Token Recall:         {statistics.mean(answer_recall):.4f}")
    print(f"Token F1:             {statistics.mean(answer_f1):.4f}")
    print(f"ROUGE-L:              {statistics.mean(answer_rouge):.4f}")

    print("\n=== RETRIEVAL QUALITY ===")
    print(
        f"Retrieved output available for "
        f"{retrieval_available_count}/{len(totals)} queries"
    )
    print(
        f"Gold passage labeled for "
        f"{retrieval_evaluable_count}/{len(totals)} queries"
    )
    if retrieval_evaluable_count:
        for k in (1, 5, 10):
            values = retrieval_metrics.get(f"hit@{k}", [])
            if values:
                print(f"Hit@{k:<13} {statistics.mean(values):.4f}")
        mrr_values = retrieval_metrics.get("mrr", [])
        if mrr_values:
            print(f"MRR:                {statistics.mean(mrr_values):.4f}")
    else:
        print("WARNING: no rows have an is_selected gold passage")

    print("\n=== LATENCY ===")
    print(f"{'stage':<28}{'avg':>10}{'p50':>10}{'p95':>10}{'p99':>10}   (ms)")
    for name, values in [*stage_ms.items(), ("total", totals)]:
        print(
            f"{name:<28}"
            f"{statistics.mean(values):>10.2f}"
            f"{percentile(values, 50):>10.2f}"
            f"{percentile(values, 95):>10.2f}"
            f"{percentile(values, 99):>10.2f}"
        )

    p95_total = percentile(totals, 95)
    print(f"\nLatency budget: {LATENCY_BUDGET_MS:.0f}ms | p95 total: {p95_total:.2f}ms")
    print("PASS" if p95_total <= LATENCY_BUDGET_MS else "FAIL")

    print(f"\nDetailed per-query report: {args.output}")

    # Deterministic quality failure signal only. Latency is reported above but
    # is intentionally NOT used as a pass/fail gate while retrieval quality
    # is being debugged.
    quality_fail = statistics.mean(answer_f1) < 0.50

    worst = sorted(
        rows_out,
        key=lambda row: float(row["answer_f1"]),
    )[:5]

    print("\n=== WORST 5 ANSWERS ===")
    for item in worst:
        print(f"\nQ: {item['query']}")
        print(f"Gold: {item['best_gold']}")
        print(f"Pred: {item['prediction']}")
        print(
            f"F1={float(item['answer_f1']):.3f} "
            f"Hit@1={float(item['retrieval'].get('hit@1', 0.0)):.0f} "
            f"Hit@5={float(item['retrieval'].get('hit@5', 0.0)):.0f} "
            f"Hit@10={float(item['retrieval'].get('hit@10', 0.0)):.0f}"
        )

    if quality_fail:
        print("\nFAIL: mean answer Token-F1 is below 0.50")
        sys.exit(1)

    print("\nQUALITY PASS")


if __name__ == "__main__":
    main()
