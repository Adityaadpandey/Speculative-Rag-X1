from speculative_rag.schemas import Chunk


def recall_hit(retrieved: list[Chunk], relevant_passage_ids: set[str], k: int) -> bool:
    top_k_source_ids = {c.source_passage_id for c in retrieved[:k]}
    return bool(top_k_source_ids & relevant_passage_ids)


def compute_recall_at_k(hits: list[bool]) -> float:
    if not hits:
        return 0.0
    return sum(hits) / len(hits)
