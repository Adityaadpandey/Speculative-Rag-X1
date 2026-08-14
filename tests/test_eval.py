from speculative_rag.eval import compute_recall_at_k, recall_hit
from speculative_rag.schemas import Chunk


def _chunk(source_passage_id, strategy="native"):
    return Chunk(
        chunk_id=f"{source_passage_id}:{strategy}:0",
        text="text",
        language="en",
        query_type="DESCRIPTION",
        source_passage_id=source_passage_id,
        strategy=strategy,
    )


def test_recall_hit_true_when_relevant_passage_in_top_k():
    retrieved = [_chunk("a"), _chunk("b"), _chunk("c")]
    assert recall_hit(retrieved, {"c"}, k=3) is True


def test_recall_hit_false_when_relevant_passage_outside_top_k():
    retrieved = [_chunk("a"), _chunk("b"), _chunk("c")]
    assert recall_hit(retrieved, {"c"}, k=2) is False


def test_recall_hit_false_when_no_relevant_passages():
    retrieved = [_chunk("a")]
    assert recall_hit(retrieved, set(), k=1) is False


def test_compute_recall_at_k_averages_hits():
    assert compute_recall_at_k([True, True, False, False]) == 0.5


def test_compute_recall_at_k_empty_is_zero():
    assert compute_recall_at_k([]) == 0.0
