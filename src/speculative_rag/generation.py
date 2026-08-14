"""Extractive answer selection: rank and return the best-matching span from
the top retrieved passages. This is a ranking operation that finishes in
single-digit milliseconds -- no model call, no free-form generation."""

from speculative_rag.chunking import split_sentences
from speculative_rag.text_utils import tokenize


def select_answer_span(query_text: str, passage_text: str) -> str:
    """Return the sentence in the passage with the most token overlap with the
    query. Falls back to the whole passage when it is a single sentence or no
    sentence shares any token with the query (span selection adds no signal)."""
    sentences = split_sentences(passage_text)
    if len(sentences) <= 1:
        return passage_text.strip()

    query_tokens = tokenize(query_text)

    def overlap(sentence: str) -> int:
        return len(tokenize(sentence) & query_tokens)

    best = max(sentences, key=overlap)
    if overlap(best) == 0:
        return passage_text.strip()
    return best.strip()
