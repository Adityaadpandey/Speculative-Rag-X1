from speculative_rag.schemas import Chunk


def top_k_distinct_passage_ids(retrieved: list[Chunk], k: int) -> set[str]:
    seen: list[str] = []
    for chunk in retrieved:
        if chunk.source_passage_id not in seen:
            seen.append(chunk.source_passage_id)
        if len(seen) >= k:
            break
    return set(seen)


def recall_hit(retrieved: list[Chunk], relevant_passage_ids: set[str], k: int) -> bool:
    top_k_source_ids = top_k_distinct_passage_ids(retrieved, k)
    return bool(top_k_source_ids & relevant_passage_ids)


def compute_recall_at_k(hits: list[bool]) -> float:
    if not hits:
        return 0.0
    return sum(hits) / len(hits)
