import re

from speculative_rag.text_utils import tokenize

# A cheap, hot-path-safe gate against verbatim-but-non-answering extraction:
# a regex/keyword classifier for the small set of query answer-types we can
# verify without a model call, plus a matching lexical check on the
# extracted span. This isn't full answer-type classification -- most
# queries stay unclassified (None) and just don't get gated.

_YESNO_PATTERN = re.compile(
    r"^(is|are|was|were|do|does|did|can|could|will|would|has|have|should)\b"
)
_NUMERIC_KEYWORDS = ("how many", "how much", "how old", "what year", "which year", "what date")
_PERSON_PATTERN = re.compile(r"^who\b")


def expected_answer_type(query_text: str) -> str | None:
    """Classify a query into a small closed set of cheaply-checkable answer
    types, or None if it doesn't match any (most DESCRIPTION-style queries
    won't, and that's fine -- we only gate the types we can cheaply verify)."""
    query = query_text.strip().lower()
    if _YESNO_PATTERN.match(query):
        return "yesno"
    if query.startswith("when") or any(keyword in query for keyword in _NUMERIC_KEYWORDS):
        return "numeric"
    if _PERSON_PATTERN.match(query):
        return "person"
    return None


def span_satisfies_expected_type(expected_type: str | None, span_text: str) -> bool:
    """True if span_text plausibly contains the kind of answer expected_type
    demands. None (unclassified) always passes -- there's no check to run,
    so don't block on it."""
    if expected_type is None:
        return True
    if expected_type == "yesno":
        return bool(tokenize(span_text) & {"yes", "no"})
    if expected_type == "numeric":
        return bool(re.search(r"\d", span_text))
    if expected_type == "person":
        # Coarse heuristic proxy for "contains a proper noun/name", not real NER.
        words = span_text.split()[1:]
        return any(word[0].isupper() for word in words if word.isalpha())
    return True
