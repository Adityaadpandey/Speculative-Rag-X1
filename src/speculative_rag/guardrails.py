import re

from speculative_rag.text_utils import tokenize

# A lightweight local keyword/regex filter for the timed hot path -- no call
# out to an external moderation API. Kept small and conservative on purpose:
# this is an explicit-refusal tripwire, not a full moderation model.
_UNSAFE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bhow to (make|build|create)\b.{0,30}\b(bomb|explosive|weapon|gun)\b", re.IGNORECASE),
    re.compile(r"\bhow to (hack|breach|break into)\b", re.IGNORECASE),
    re.compile(r"\b(kill|murder|poison)\b.{0,20}\b(someone|somebody|a person|people|him|her|them)\b", re.IGNORECASE),
    re.compile(r"\bchild\b.{0,15}\b(porn|sexual|abuse)\b", re.IGNORECASE),
    re.compile(r"\bhow to\b.{0,20}\bsuicide\b", re.IGNORECASE),
]


def check_unsafe_input(query_text: str) -> str | None:
    """Return a typed refusal reason if the query trips an unsafe pattern,
    else None. Runs before embedding so unsafe input never touches the index."""
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(query_text):
            return f"query matched unsafe-input pattern /{pattern.pattern}/"
    return None


def is_off_topic(top_score: float, threshold: float) -> bool:
    """Off-topic when the best retrieval score is below threshold -- the corpus
    just doesn't cover this query. Reuses the score already computed at
    retrieval.

    With BM25 sparse retrieval (unbounded scores, no fixed max), a positive
    score does NOT mean "topically relevant" -- one incidentally shared word
    between the query and an unrelated passage is enough to produce a
    positive score regardless of real relevance (I've actually seen an
    off-topic query outscore a relevant one). The only score value that
    reliably means "not covered by the corpus" is exactly 0 -- zero shared
    vocabulary with anything indexed. So off_topic_threshold is a near-zero
    floor, not a relevance cutoff; anything above it gets passed through to
    the downstream extractive/groundedness/answer-type guardrails, which are
    what actually catch a bad-but-nonzero-score match."""
    return top_score < threshold


def groundedness_score(answer_text: str, source_text: str) -> float:
    """Fraction of the answer's tokens present in the source passage. For
    extractive answers this is ~1.0 by construction; a low value means the
    generated text drifted from its source. Lexical, no model call."""
    answer_tokens = tokenize(answer_text)
    if not answer_tokens:
        return 0.0
    source_tokens = tokenize(source_text)
    return len(answer_tokens & source_tokens) / len(answer_tokens)
