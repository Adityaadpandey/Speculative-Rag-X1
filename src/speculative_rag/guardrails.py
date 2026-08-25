import re

from speculative_rag.scoring import content_query_coverage
from speculative_rag.text_utils import tokenize, tokenize_ordered

_QUESTION_LEAD = re.compile(r"^(what|which|who|where|when|why|how)\b", re.IGNORECASE)
_ANSWER_VERB = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|can|could|will|would|should|"
    r"do|does|did|make|makes|made|use|uses|used|include|includes|including|"
    r"require|requires|required|cause|causes|cost|costs|range|ranges|"
    r"located|situated|founded|called|known|means|refer|refers|"
    r"provide|provides|provided|contain|contains|formed|affect|affects)\b",
    re.IGNORECASE,
)


def _reverses_simple_direction(query_text: str, answer_text: str) -> bool:
    """Catch answers that reverse a query's explicit ``A to B`` relation."""
    query = tokenize_ordered(query_text)
    answer = tokenize_ordered(answer_text)
    try:
        pivot = query.index("to")
    except ValueError:
        return False
    before = next((token for token in reversed(query[:pivot]) if token.isalpha()), None)
    after = next((token for token in query[pivot + 1 :] if token.isalpha()), None)
    if before is None or after is None:
        return False
    try:
        before_at = answer.index(before)
        after_at = answer.index(after)
    except ValueError:
        return False
    return after_at < before_at

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


def plausible_extractive_answer(
    query_text: str,
    answer_text: str,
    matched_tokens: list[str],
    language: str,
    *,
    min_query_coverage: float = 0.60,
) -> bool:
    """Reject obvious snippets that are topical but do not answer a query.

    This is intentionally conservative and model-free: require most
    meaningful query terms, reject a retrieved question/heading, and reject
    very short noun fragments with no predicate.  It prevents a passage that
    merely repeats the query (or shares one topical word) from becoming a
    confident answer while keeping the hot path sub-millisecond.
    """
    text = answer_text.strip()
    if not text or "?" in text or _QUESTION_LEAD.match(text):
        return False
    if _reverses_simple_direction(query_text, text):
        return False
    coverage = content_query_coverage(tokenize_ordered(query_text), matched_tokens, language)
    if coverage < min_query_coverage:
        return False
    words = tokenize(text)
    if len(words) <= 5 and not _ANSWER_VERB.search(text):
        return False
    return True


def groundedness_score(answer_text: str, source_text: str) -> float:
    """Fraction of the answer's tokens present in the source passage. For
    extractive answers this is ~1.0 by construction; a low value means the
    generated text drifted from its source. Lexical, no model call."""
    answer_tokens = tokenize(answer_text)
    if not answer_tokens:
        return 0.0
    source_tokens = tokenize(source_text)
    return len(answer_tokens & source_tokens) / len(answer_tokens)
