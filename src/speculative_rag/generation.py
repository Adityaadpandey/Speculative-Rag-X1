"""Extractive answer selection: rank and return the best-matching span from
the top retrieved passages. This is a ranking operation that finishes in
single-digit milliseconds -- no model call, no free-form generation."""

from speculative_rag.answer_type import expected_answer_type, span_satisfies_expected_type
from speculative_rag.chunking import split_sentences
from speculative_rag.pipeline_schemas import ScoredChunk
from speculative_rag.text_utils import tokenize


def select_answer_span(query_text: str, passage_text: str) -> str | None:
    """Return the sentence in the passage with the most token overlap with the
    query. Falls back to the whole passage only when it's a single sentence
    (there's nothing better to return). When there are multiple sentences but
    none of them share a token with the query, that's a signal the passage
    doesn't actually contain the answer -- return None instead of dumping the
    full passage as if it were.

    Among sentences that do share vocabulary with the query, a sentence that
    just mentions the same topic can out-overlap the one that actually holds
    the answer (a query asking for a credit score can match a sentence about
    "good credit" better, lexically, than the one with "660" in it). So when
    the query has a cheaply-checkable expected answer type (numeric, yesno,
    person -- see answer_type.py), narrow the candidates to sentences that
    satisfy it before ranking by overlap, so a type-appropriate sentence wins
    over a merely-topical one. Falls back to the full overlap-ranked set when
    nothing satisfies the expected type, since that check is a coarse
    heuristic and may just not apply here.

    Ties on raw overlap do happen -- an intro sentence that restates the
    passage's own subject verbatim vs. the sentence that actually answers an
    "impact"/"result"-of-X query in different words -- and overlap alone
    can't break them since neither sentence stands out lexically. We break
    ties toward the later sentence: setup-then-consequence is the dominant
    ordering for this kind of query, so the later sentence is the better
    default guess when overlap can't decide."""
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

    best = max(candidates, key=lambda s: (overlap(s), sentences.index(s)))
    return best.strip()


def select_best_evidence(
    query_text: str, candidates: list[ScoredChunk]
) -> tuple[ScoredChunk, str] | None:
    """A fast, model-free "rerank": instead of extracting only from embedding
    rank-1, run select_answer_span across each of the top candidates (already
    scored and sorted by the retriever, so this costs no extra retrieval) and
    pick the best (passage, span) pair. Fixes the case where rank-1 is
    on-topic but never actually states the answer -- e.g. a passage
    describing a project's purpose when the query asks about its outcome --
    while a lower-ranked passage does.

    Ranked by (does an expected-answer-type sentence exist in this passage,
    token overlap with the query, retrieval score) -- the same signals
    select_answer_span already uses for one passage, just extended across
    several. Returns None only when no candidate has any sentence sharing
    vocabulary with the query at all."""
    query_tokens = tokenize(query_text)
    expected_type = expected_answer_type(query_text)

    best: tuple[ScoredChunk, str] | None = None
    best_key: tuple[bool, int, float] | None = None
    for candidate in candidates:
        span = select_answer_span(query_text, candidate.chunk.text)
        if span is None:
            continue
        type_match = expected_type is None or span_satisfies_expected_type(expected_type, span)
        overlap_count = len(tokenize(span) & query_tokens)
        key = (type_match, overlap_count, candidate.score)
        if best_key is None or key > best_key:
            best_key = key
            best = (candidate, span)
    return best
