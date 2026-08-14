import unicodedata

# A "word" character is any Unicode letter or digit (general category L* or
# N*) plus any combining mark (category Mn = nonspacing, Mc = spacing
# combining) that follows one. Combining marks are how scripts like
# Devanagari and Tamil attach vowel signs/virama to a base letter, so
# excluding them would silently split a single word into fragments.
# Scanning by unicodedata.category() instead of hardcoding per-script
# codepoint ranges makes this work for any script without having to
# enumerate ranges one by one.
_WORD_CATEGORIES = ("L", "N")
_COMBINING_CATEGORIES = ("Mn", "Mc")


def _is_word_char(ch: str) -> bool:
    category = unicodedata.category(ch)
    return category[0] in _WORD_CATEGORIES or category in _COMBINING_CATEGORIES


def tokenize(text: str) -> set[str]:
    """Lowercased set of word tokens. Unicode-aware so it works for
    Devanagari/Tamil as well as Latin scripts -- used by both the
    groundedness check and extractive span selection."""
    tokens = []
    current: list[str] = []
    for ch in text:
        if _is_word_char(ch):
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return {token.lower() for token in tokens}
