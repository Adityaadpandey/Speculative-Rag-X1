"""Percentile math for the RAG-only latency report.
P100 is just the single worst observed value; P50/P70 are what actually
describe typical behaviour."""


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if pct >= 100:
        return ordered[-1]
    if pct <= 0:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 50),
        "p70": percentile(values, 70),
        "p100": percentile(values, 100),
    }
