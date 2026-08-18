"""
Deterministic RAG quality + latency benchmark.

NO LLM / NO LLM-AS-JUDGE:
- Answer Exact Match
- Token Precision / Recall / F1
- ROUGE-L (token LCS)
- Retrieval Hit@K / MRR when retrieved passages are exposed
- Latency per pipeline stage + end-to-end latency
- Per-query JSONL report for debugging

Expected dataset schema (MS MARCO / MSMARCO-XI style):
    query: str
    answers: list[str]
    passages: {
        is_selected: list[int],
        passage_text: list[str],
        url: list[str]
    }

Usage:
    uv run python -m scripts.benchmark_quality path/to/validation.parquet
    uv run python -m scripts.benchmark_quality path/to/validation.parquet 100
    uv run python -m scripts.benchmark_quality path/to/validation.parquet 100 hi

Optional:
    --output results.jsonl
    --top-k 10

Notes:
- This benchmark evaluates your actual run_pipeline() output.
- It never calls an LLM to decide whether an answer is good.
- Answer matching is lexical, so paraphrases can score lower even when humans
  would consider them correct. That is intentional: the benchmark is
  deterministic and reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from speculative_rag.config import SETTINGS, get_qdrant_client
from speculative_rag.latency import percentile
from speculative_rag.pipeline import run_pipeline
from speculative_rag.retrieval import Retriever
from speculative_rag.sparse_embedding import SparseEmbedder


LATENCY_BUDGET_MS = 200.0

DEFAULT_TOP_K = 10

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


def get_answers(row: dict[str, Any]) -> list[str]:
    answers = row.get("answers", [])
    if answers is None:
        return []
    if isinstance(answers, str):
        return [answers]
    return [str(x) for x in answers if x is not None]


def get_gold_passages(row: dict[str, Any]) -> list[str]:
    passages = row.get("passages", {})
    if not isinstance(passages, dict):
        return []

    texts = passages.get("passage_text", []) or []
    selected = passages.get("is_selected", []) or []

    out = []
    for text, flag in zip(texts, selected):
        try:
            chosen = int(flag) == 1
        except (TypeError, ValueError):
            chosen = bool(flag)
        if chosen:
            out.append(str(text))
    return out


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


def extract_answer(result: Any) -> str:
    value = first_attr(result, (
        "answer",
        "final_answer",
        "response",
        "text",
        "generated_answer",
    ))
    return "" if value is None else str(value)


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


def extract_retrieved_passages(result: Any) -> list[str]:
    candidates = first_attr(result, (
        "retrieved_passages",
        "retrieved_documents",
        "retrieved_docs",
        "contexts",
        "documents",
        "passages",
        "candidates",
        "hits",
        "sources",
        "results",
    ))

    return flatten_passage_candidates(candidates)


def retrieval_rank_metrics(
    retrieved_passages: Sequence[str],
    gold_passages: Sequence[str],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    if not gold_passages or not retrieved_passages:
        return {
            **{f"hit@{k}": 0.0 for k in ks},
            "mrr": 0.0,
            "retrieval_available": 0.0,
        }

    gold_norm = {
        normalize_passage_for_match(p)
        for p in gold_passages
        if normalize_passage_for_match(p)
    }

    first_rank = None
    for idx, passage in enumerate(retrieved_passages, start=1):
        if normalize_passage_for_match(passage) in gold_norm:
            first_rank = idx
            break

    metrics = {
        f"hit@{k}": float(first_rank is not None and first_rank <= k)
        for k in ks
    }
    metrics["mrr"] = 1.0 / first_rank if first_rank is not None else 0.0
    metrics["retrieval_available"] = 1.0
    return metrics


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

    rows = load_rows(args.dataset)
    if not rows:
        raise SystemExit("Dataset is empty.")

    n = min(args.n_queries, len(rows))
    rows = rows[:n]

    collection_name = f"{args.language}_native"

    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        raise SystemExit(
            f"No such collection '{collection_name}'. Available: {available}\n"
            "Build the collection first."
        )

    embedder = SparseEmbedder(args.language)
    retriever = Retriever(client, collection_name)

    print(f"Benchmarking {n} queries against '{collection_name}'")
    print("No LLM evaluator is used. Answer scoring is deterministic.\n")

    # Warm up model / DB connection.
    warmup_query = str(rows[0]["query"])
    run_pipeline(
        warmup_query,
        embedder,
        retriever,
        off_topic_threshold=SETTINGS.off_topic_threshold,
        groundedness_threshold=SETTINGS.groundedness_threshold,
        top_k=args.top_k,
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

    for i, row in enumerate(rows, start=1):
        query = str(row.get("query", "")).strip()
        if not query:
            continue

        gold_answers = get_answers(row)
        gold_passages = get_gold_passages(row)

        result = run_pipeline(
            query,
            embedder,
            retriever,
            off_topic_threshold=SETTINGS.off_topic_threshold,
            groundedness_threshold=SETTINGS.groundedness_threshold,
            top_k=args.top_k,
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

        retrieved_passages = extract_retrieved_passages(result)
        rmetrics = retrieval_rank_metrics(retrieved_passages, gold_passages)

        if rmetrics["retrieval_available"]:
            retrieval_available_count += 1

        for name, value in rmetrics.items():
            if name != "retrieval_available":
                retrieval_metrics[name].append(value)

        row_out = {
            "index": i,
            "query_id": row.get("query_id"),
            "query": query,
            "prediction": prediction,
            "gold_answers": gold_answers,
            "best_gold": answer_score["gold"],
            "answer_exact_match": answer_score["exact_match"],
            "answer_precision": answer_score["precision"],
            "answer_recall": answer_score["recall"],
            "answer_f1": answer_score["f1"],
            "answer_rouge_l": answer_score["rouge_l"],
            "gold_selected_passage_count": len(gold_passages),
            "retrieved_passage_count": len(retrieved_passages),
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
    if retrieval_available_count:
        for k in (1, 5, 10):
            values = retrieval_metrics.get(f"hit@{k}", [])
            if values:
                print(f"Hit@{k:<13} {statistics.mean(values):.4f}")
        mrr_values = retrieval_metrics.get("mrr", [])
        if mrr_values:
            print(f"MRR:                {statistics.mean(mrr_values):.4f}")
        print(
            f"Retrieved output available for "
            f"{retrieval_available_count}/{len(totals)} queries"
        )
    else:
        print(
            "WARNING: run_pipeline() did not expose retrieved passages, "
            "so Hit@K/MRR could not be computed."
        )
        print(
            "Expose a field such as `retrieved_passages` / `contexts` on the "
            "PipelineResult if you want retrieval metrics."
        )

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

    # A useful hard regression signal.
    quality_fail = statistics.mean(answer_f1) < 0.50
    latency_fail = p95_total > LATENCY_BUDGET_MS

    if quality_fail:
        print("FAIL: mean answer Token-F1 is below 0.50")
    if latency_fail:
        print("FAIL: p95 latency exceeds 200ms")

    if quality_fail or latency_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
