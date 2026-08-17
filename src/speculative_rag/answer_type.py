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
# Query-language keyword matching only works per-script -- English words
# never show up as substrings of Hindi/Tamil query text and vice versa -- so
# this corpus (hi/en/ta, see config.LANGUAGE_SOURCES) needs each language's
# own keyword set, not just an English one run against translated text.
# There's no yesno set for hi/ta: unlike English's clean auxiliary-verb
# prefix, Hindi/Tamil yes-no questions are usually marked by sentence-final
# particles or intonation, not a reliable sentence-initial token. A naive
# prefix guess (Hindi "क्या", say) is genuinely ambiguous with the "what"
# interrogative -- I checked real data and "क्या गैस प्रोग्राम है" is a
# DESCRIPTION query, not yesno -- so that guess would misclassify more than
# it fixes.
_NUMERIC_KEYWORDS = (
    # English
    "how many", "how much", "how old", "what year", "which year", "what date",
    "weight", "height", "score", "temperature", "price", "cost", "distance",
    "population", "percentage", "speed", "size", "length", "width", "depth",
    # Hindi
    "कितने", "कितना", "कितनी", "किस वर्ष", "कौन से वर्ष", "कौन सी तारीख",
    "वजन", "स्कोर", "तापमान", "कीमत", "मूल्य", "दूरी", "जनसंख्या", "प्रतिशत",
    "गति", "आकार", "लंबाई", "चौड़ाई", "गहराई",
    # Tamil
    "எத்தனை", "எவ்வளவு", "எந்த ஆண்டு", "வயது", "எடை", "மதிப்பெண்", "ஸ்கோர்",
    "விலை", "தூரம்", "மக்கள் தொகை", "சதவீதம்", "வேகம்", "நீளம்", "அகலம்", "ஆழம்", "வெப்பநிலை",
)
_NUMERIC_START_WORDS = ("when", "कब", "எப்போது")
# Deliberately not \b: regex \b treats a trailing combining mark (e.g. Tamil
# "யார்"'s virama, U+0BCD, category Mn) as a non-word character, so "\b"
# right after it never counts as a boundary and the match silently fails.
# (?=\s|$) sidesteps that by checking literally for whitespace-or-end
# instead of a \w/\W transition.
_PERSON_PATTERN = re.compile(r"^(who|कौन|யார்)(?=\s|$)")


def expected_answer_type(query_text: str) -> str | None:
    """Classify a query into a small closed set of cheaply-checkable answer
    types, or None if it doesn't match any (most DESCRIPTION-style queries
    won't, and that's fine -- we only gate the types we can cheaply verify)."""
    query = query_text.strip().lower()
    if _YESNO_PATTERN.match(query):
        return "yesno"
    if any(query.startswith(w) for w in _NUMERIC_START_WORDS) or any(
        keyword in query for keyword in _NUMERIC_KEYWORDS
    ):
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
        # A coarse heuristic proxy for "contains a proper noun/name," not
        # real NER -- and only meaningful for scripts with a case
        # distinction. Devanagari/Tamil have no letter-casing, so
        # word[0].isupper() is False for every word regardless of content;
        # treating that as a negative signal would silently reject every
        # Hindi/Tamil span. Only apply the capitalization check when the
        # span actually has Latin letters to check; otherwise this type
        # isn't verifiable here, so don't block on it (same "can't check it"
        # stance as None).
        words = span_text.split()[1:]
        latin_words = [w for w in words if w.isalpha() and w.isascii()]
        if not latin_words:
            return True
        return any(word[0].isupper() for word in latin_words)
    return True
