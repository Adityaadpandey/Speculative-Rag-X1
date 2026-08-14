from speculative_rag.latency import percentile, summarize


def test_percentile_median_of_odd_list():
    assert percentile([1, 2, 3, 4, 5], 50) == 3


def test_percentile_p100_is_max():
    assert percentile([1, 5, 2, 9, 3], 100) == 9


def test_percentile_empty_is_zero():
    assert percentile([], 50) == 0.0


def test_percentile_interpolates():
    # 50th pct of 1..10: rank = 0.5*(10-1) = 4.5 -> 5 + 0.5*(6-5) = 5.5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5.5


def test_summarize_returns_ordered_p50_p70_p100():
    summary = summarize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert summary["p100"] == 10
    assert summary["p50"] <= summary["p70"] <= summary["p100"]
