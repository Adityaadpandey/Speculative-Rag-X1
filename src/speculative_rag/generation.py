"""Extractive answer selection: rank and return the best-matching span from
the top retrieved passages. This is a ranking operation that finishes in
single-digit milliseconds -- no model call, no free-form generation."""

from speculative_rag.answer_type import expected_answer_type, span_satisfies_expected_type
from speculative_rag.chunking import split_sentences
from speculative_rag.text_utils import tokenize


def select_answer_span(query_text: str, passage_text: str) -> str | None:
    """Return the sentence in the passage with the most token overlap with the
    query. Falls back to the whole passage only when it's a single sentence
    (there's nothing better to return). When there are multiple sentences but
    none of them share a token with the query, that's a signal the passage
    doesn't actually contain the answer -- return None instead of dumping the
    full passage as if it were.

    Among sentences that do share vocabulary with the query, a sentence that
    merely mentions the same topic can out-overlap the sentence that actually
    holds the answer (e.g. a query asking for a credit score matching a
    sentence about "good credit" better, lexically, than the one with "660").
    When the query has a cheaply-checkable expected answer type (numeric,
    yesno, person — see answer_type.py), narrow the candidates to sentences
    that satisfy it before ranking by overlap, so a type-appropriate sentence
    wins over a merely-topical one. Falls back to the full overlap-ranked set
    when no candidate satisfies the expected type, since the type check is a
    coarse heuristic and may simply not apply here."""
    sentences = split_sentences(passage_text)
    if len(sentences) <= 1:
        return passage_text.strip()

    query_tokens = tokenize(query_text)

    def overlap(sentence: str) -> int:
        return len(tokenize(sentence) & query_tokens)

    candidates = [s for s in sentences if overlap(s) > 0]
    if not candidates:
        return None

    expected_type = expected_answer_type(query_text)
    if expected_type is not None:
        typed_candidates = [s for s in candidates if span_satisfies_expected_type(expected_type, s)]
        if typed_candidates:
            candidates = typed_candidates

    return max(candidates, key=overlap).strip()
