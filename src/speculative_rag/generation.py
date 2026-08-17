"""Extractive answer selection: rank and return the best-matching span from
the top retrieved passages. This is a ranking operation that finishes in
single-digit milliseconds -- no model call, no free-form generation."""

from speculative_rag.chunking import split_sentences
from speculative_rag.text_utils import tokenize


def select_answer_span(query_text: str, passage_text: str) -> str | None:
    """Return the sentence in the passage with the most token overlap with the
    query. Falls back to the whole passage only when it's a single sentence
    (there's nothing better to return). When there are multiple sentences but
    none of them share a token with the query, that's a signal the passage
    doesn't actually contain the answer -- return None instead of dumping the
    full passage as if it were."""
    sentences = split_sentences(passage_text)
    if len(sentences) <= 1:
        return passage_text.strip()

    query_tokens = tokenize(query_text)

    def overlap(sentence: str) -> int:
        return len(tokenize(sentence) & query_tokens)

    best = max(sentences, key=overlap)
    if overlap(best) == 0:
        return None
    return best.strip()
