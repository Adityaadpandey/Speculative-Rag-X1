import re

# Match word characters (letters, digits) followed by optional combining marks
# (for Devanagari, Tamil, etc.). Combining marks include diacritics, stress marks.
# Pattern: base char + optional (word char or combining mark)
_TOKEN_RE = re.compile(
    r"[^\W_](?:[^\W_]|[̀-ͯ҃-҉ँ-॔ॢ-ॣ])*",
    re.UNICODE
)


def tokenize(text: str) -> set[str]:
    """Lowercased set of word tokens. Unicode-aware so it works for
    Devanagari/Tamil as well as Latin scripts -- used by both the
    groundedness check and extractive span selection."""
    return {match.lower() for match in _TOKEN_RE.findall(text)}
