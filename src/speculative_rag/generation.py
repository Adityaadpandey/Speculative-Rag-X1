"""Extractive answer selection: rank and return the best-matching span from
the top retrieved passages. This is a ranking operation that finishes in
single-digit milliseconds -- no model call, no free-form generation."""

import re
from typing import NamedTuple

from speculative_rag.answer_type import expected_answer_type, span_satisfies_expected_type
from speculative_rag.chunking import split_sentences
from speculative_rag.pipeline_schemas import ScoredChunk
from speculative_rag.scoring import ScoredSentence, score_sentence
from speculative_rag.text_utils import tokenize_ordered

# Some passages in the source corpus repeat their own opening text later in
# the same passage -- glued on with no separating whitespace and missing its
# first character (e.g. a passage whose text is "The most visited ...
# Germany." followed immediately, mid-word, by "he most visited ...
# Germany." again). That's a corpus-ingestion artifact, not real content,
# and it corrupts extractive answers directly: without stripping it,
# select_answer_span can hand back a sentence that's literally two copies of
# itself stitched together.
_DUPLICATE_TAIL_PROBE_LEN = 30

# Comma/semicolon clause boundaries, used to narrow a winning sentence down
# to just the clause that actually satisfies an expected answer type -- see
# _trim_to_best_clause.
_CLAUSE_SPLIT_RE = re.compile(r"[,;]")

# Leading connective phrases stripped from the final span so it reads as a
# standalone sentence rather than a mid-argument fragment ("However, the
# launch was delayed." -> "The launch was delayed."). English/Latin-script
# only, same scope as the abbreviation list in chunking.py.
_LEADING_CONNECTIVES = (
    "However, ", "In addition, ", "Additionally, ", "Furthermore, ", "Moreover, ",
    "Meanwhile, ", "Also, ", "Therefore, ", "Thus, ", "Consequently, ",
)


def _strip_duplicate_tail(text: str) -> str:
    """Look for a long, exact repeat of the passage's own opening (dropping
    its first character, which is the corruption pattern we've actually
    seen -- or, failing that, an exact repeat with no character dropped) and
    cut the passage off right before it. The probe is long enough that an
    accidental match on ordinary prose is basically impossible, so this only
    ever fires on the real corruption."""
    if len(text) < _DUPLICATE_TAIL_PROBE_LEN * 3:
        return text
    for drop in (1, 0):
        probe = text[drop : drop + _DUPLICATE_TAIL_PROBE_LEN]
        idx = text.find(probe, _DUPLICATE_TAIL_PROBE_LEN)
        if idx != -1:
            return text[:idx].rstrip()
    return text


def _uppercase_first_letter(text: str) -> str:
    """Uppercase only the first cased character, leaving everything else
    untouched. Deliberately not str.capitalize(): capitalize() lowercases
    every OTHER character in the string too, which corrupts mid-sentence
    proper nouns/acronyms ("NASA launched..." -> "Nasa launched..."). A
    no-op for scripts with no letter casing (Devanagari, Tamil), since
    ch.isalpha() combined with ch.upper() being a no-op covers that."""
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1 :]
    return text


def _strip_leading_connective(text: str) -> str:
    for connective in _LEADING_CONNECTIVES:
        if text.startswith(connective):
            return _uppercase_first_letter(text[len(connective) :])
    return text


def _trim_to_best_clause(
    sentence: str, expected_type: str | None, query_tokens: list[str], language: str
) -> str:
    """When the query has a cheaply-checkable expected answer type and the
    winning sentence has multiple comma/semicolon-delimited clauses, narrow
    the returned span to just the clause that actually satisfies the type --
    plain type-matching alone is unsafe here, since a sentence can contain a
    relevant-looking number in a clause that has nothing to do with the
    query (e.g. a year mentioned in an unrelated aside).

    Selection order among type-satisfying clauses: weighted content score,
    then the bigram/phrase-score component alone as a tiebreak within score
    ties, then original clause position (later wins) as the final tiebreak
    -- kept consistent with the later-sentence-wins convention used for
    whole-sentence selection above."""
    if expected_type is None:
        return sentence

    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(sentence) if c.strip()]
    if len(clauses) <= 1:
        return sentence

    typed = [c for c in clauses if span_satisfies_expected_type(expected_type, c)]
    if not typed:
        return sentence

    def clause_key(clause: str) -> tuple[float, float, int]:
        scored = score_sentence(query_tokens, tokenize_ordered(clause), language)
        return (scored.score, scored.bigram_score, clauses.index(clause))

    return max(typed, key=clause_key)


class _SelectedSpan(NamedTuple):
    text: str  # clause-trimmed, user-facing span
    raw_sentence: str  # untrimmed winning sentence, for provenance
    matched_tokens: list[str]


def _select_span_from_passage(query_text: str, passage_text: str, language: str) -> _SelectedSpan | None:
    """Core span-selection logic shared by select_answer_span (single
    passage) and select_best_evidence (best of several passages). Returns
    None when nothing in the passage shares *meaningful* (content-word,
    non-stopword) vocabulary with the query -- a stopword-only match (e.g.
    sharing only "the", "is", "of") is not evidence, and dumping the whole
    passage back in that case would misrepresent it as an answer."""
    passage_text = _strip_duplicate_tail(passage_text)
    sentences = split_sentences(passage_text)
    query_tokens = tokenize_ordered(query_text)

    token_cache: dict[str, list[str]] = {}

    def sentence_tokens(sentence: str) -> list[str]:
        cached = token_cache.get(sentence)
        if cached is None:
            cached = tokenize_ordered(sentence)
            token_cache[sentence] = cached
        return cached

    def score(sentence: str) -> ScoredSentence:
        return score_sentence(query_tokens, sentence_tokens(sentence), language)

    expected_type = expected_answer_type(query_text)

    if len(sentences) <= 1:
        candidate_text = passage_text.strip()
        candidate_score = score(candidate_text)
        if candidate_score.content_overlap == 0:
            return None
        winner, winner_score = candidate_text, candidate_score
    else:
        scored = {s: score(s) for s in sentences}
        candidates = [s for s in sentences if scored[s].content_overlap > 0]
        if not candidates:
            return None

        # A cheaply-checkable expected answer type (numeric, yesno, person --
        # see answer_type.py) narrows candidates to type-satisfying sentences
        # before ranking, so a type-appropriate sentence wins over a
        # merely-topical one (a query asking for a credit score can match a
        # sentence about "good credit" better, lexically, than the one with
        # "660" in it). Falls back to the full set when nothing satisfies the
        # expected type, since that check is a coarse heuristic and may not
        # apply here.
        if expected_type is not None:
            typed_candidates = [s for s in candidates if span_satisfies_expected_type(expected_type, s)]
            if typed_candidates:
                candidates = typed_candidates

        # Ties on raw score do happen and can't be broken by score alone --
        # we break toward the later sentence: setup-then-consequence is the
        # dominant ordering for this kind of query, so the later sentence is
        # the better default guess when score can't decide.
        winner = max(candidates, key=lambda s: (scored[s].score, sentences.index(s)))
        winner_score = scored[winner]

    trimmed = _trim_to_best_clause(winner, expected_type, query_tokens, language)
    trimmed = _strip_leading_connective(trimmed)
    return _SelectedSpan(text=trimmed, raw_sentence=winner.strip(), matched_tokens=winner_score.matched_tokens)


def select_answer_span(query_text: str, passage_text: str, language: str) -> str | None:
    """Return the best-matching span in the passage for the query, or None
    when nothing in the passage shares meaningful vocabulary with it. See
    _select_span_from_passage for the full selection/trimming logic."""
    result = _select_span_from_passage(query_text, passage_text, language)
    return result.text if result is not None else None


def select_best_evidence(
    query_text: str, candidates: list[ScoredChunk]
) -> tuple[ScoredChunk, str, str, list[str]] | None:
    """A fast, model-free "rerank": instead of extracting only from embedding
    rank-1, run span selection across each of the top candidates (already
    scored and sorted by the retriever, so this costs no extra retrieval) and
    pick the best (passage, span) pair. Fixes the case where rank-1 is
    on-topic but never actually states the answer -- e.g. a passage
    describing a project's purpose when the query asks about its outcome --
    while a lower-ranked passage does.

    Ranked by (does a type-appropriate span exist in this passage, weighted
    content score of the untrimmed winning sentence, retrieval score) -- the
    same signals _select_span_from_passage already uses for one passage,
    just extended across several. Ranking uses the untrimmed sentence's
    score rather than the clause-trimmed span's, so a heavily-trimmed short
    clause from one passage isn't penalized against an untrimmed sentence
    from another for losing surrounding context it was never meant to keep.

    Returns (chunk, clause-trimmed text, raw untrimmed sentence, matched
    query tokens) -- raw_sentence and matched_tokens carry through to
    Answer's raw_sentence/matched_tokens fields (see pipeline_schemas.py).
    Returns None only when no candidate has any sentence sharing meaningful
    vocabulary with the query at all."""
    query_tokens = tokenize_ordered(query_text)
    expected_type = expected_answer_type(query_text)

    best: tuple[ScoredChunk, str, str, list[str]] | None = None
    best_key: tuple[bool, float, float] | None = None
    for candidate in candidates:
        result = _select_span_from_passage(query_text, candidate.chunk.text, candidate.chunk.language)
        if result is None:
            continue
        type_match = expected_type is None or span_satisfies_expected_type(expected_type, result.text)
        raw_score = score_sentence(query_tokens, tokenize_ordered(result.raw_sentence), candidate.chunk.language)
        key = (type_match, raw_score.score, candidate.score)
        if best_key is None or key > best_key:
            best_key = key
            best = (candidate, result.text, result.raw_sentence, result.matched_tokens)
    return best


def prefilter_rerank_candidates(
    query_text: str,
    candidates: list[ScoredChunk],
    *,
    lexical_top_n: int,
    retrieval_reserve: int,
) -> list[ScoredChunk]:
    """Non-LLM prefilter narrowing the candidate pool handed to the async
    rerank+regenerate LLM call (answer_orchestrator.resolve_recovery). A
    candidate survives if it satisfies ANY of: top-N by weighted lexical
    score, top-N by original retrieval rank regardless of lexical score, or
    it contains a sentence satisfying the query's expected answer type.

    Deliberately a union of independent membership tests, not a single
    ranked cutoff -- reflect.py's own docstring documents that a prior cheap
    lexical-reranking attempt made recall WORSE, not better, because the
    corpus's gold passage often paraphrases the query differently than the
    verbatim-matching distractors do. Keeping the top-N by *retrieval* rank
    regardless of lexical score is the hedge against exactly that failure
    mode. `candidates` must already be in retrieval-rank order (the order
    retrieval returns them in) for the retrieval_reserve reserve to mean
    anything.

    If the entire pool has zero meaningful (non-stopword) content overlap
    with the query, skip filtering entirely and return the pool unfiltered
    -- the prefilter must never be the reason a query fails."""
    if not candidates:
        return candidates

    query_tokens = tokenize_ordered(query_text)
    expected_type = expected_answer_type(query_text)

    def content_overlap(candidate: ScoredChunk) -> int:
        return max(
            (
                score_sentence(query_tokens, tokenize_ordered(sentence), candidate.chunk.language).content_overlap
                for sentence in split_sentences(candidate.chunk.text)
            ),
            default=0,
        )

    overlaps = {c.chunk.chunk_id: content_overlap(c) for c in candidates}
    if all(overlap == 0 for overlap in overlaps.values()):
        return candidates

    by_lexical = sorted(candidates, key=lambda c: overlaps[c.chunk.chunk_id], reverse=True)
    lexical_top_ids = {c.chunk.chunk_id for c in by_lexical[:lexical_top_n]}
    retrieval_top_ids = {c.chunk.chunk_id for c in candidates[:retrieval_reserve]}

    def type_satisfied(candidate: ScoredChunk) -> bool:
        if expected_type is None:
            return False
        return any(
            span_satisfies_expected_type(expected_type, sentence) for sentence in split_sentences(candidate.chunk.text)
        )

    return [
        c
        for c in candidates
        if c.chunk.chunk_id in lexical_top_ids or c.chunk.chunk_id in retrieval_top_ids or type_satisfied(c)
    ]
