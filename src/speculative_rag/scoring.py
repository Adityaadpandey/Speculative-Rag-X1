"""Shared, ordered, language-aware relevance scoring, used by extractive
span selection (generation.py) and the recovery rerank prefilter
(generation.prefilter_rerank_candidates, called from answer_orchestrator.py).
Centralizing this in one place avoids two independently-tuned notions of
"does this sentence look relevant to the query" drifting apart.

Unlike text_utils.tokenize() (an unordered set), this scores against the
ordered token list so adjacent-word phrase matches can be rewarded, and
weights content words above stopwords so a sentence that only shares "the"
or "is" with the query doesn't count as evidence."""

from dataclasses import dataclass

# Small, closed per-language stopword lists -- same per-language-explicit
# pattern answer_type.py already uses for its keyword sets, not run through
# any stemmer or external library.
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "of", "in", "on", "at", "to", "for", "and", "or", "but", "with", "as",
            "by", "from", "that", "this", "these", "those", "it", "its", "do",
            "does", "did", "has", "have", "had", "will", "would", "can", "could",
            "should", "what", "which", "who", "when", "where", "why", "how",
            "not", "no", "if", "than",
        }
    ),
    "hi": frozenset(
        {
            "है", "हैं", "था", "थी", "थे", "का", "की", "के", "में", "पर", "से",
            "को", "और", "या", "यह", "वह", "क्या", "कौन", "कब", "कहाँ", "क्यों",
            "कैसे", "एक", "भी", "ही", "तो", "हो", "न", "नहीं", "लिए", "साथ",
        }
    ),
    "ta": frozenset(
        {
            "இது", "அது", "ஆகும்", "உள்ளது", "மற்றும்", "அல்லது", "ஒரு", "இந்த",
            "அந்த", "என்ன", "யார்", "எப்போது", "எங்கே", "ஏன்", "எப்படி",
            "இருந்து", "இல்", "மேல்", "உடன்", "ஆனால்",
        }
    ),
}

_CONTENT_WEIGHT = 1.0
_STOPWORD_WEIGHT = 0.3
# Calibrated, not hand-picked: scripts/calibrate_bigram_bonus.py swept
# {0.0, 0.2, 0.35, 0.5, 0.75, 1.0} against the en_native validation sample's
# answer-quality F1 (295 queries with a retrievable top passage). Result was
# monotonically decreasing from 0.0 (mean F1 0.1616) to 1.0 (mean F1 0.1598)
# -- on this corpus, rewarding adjacent phrase matches doesn't help and
# mildly hurts, so the bonus that actually ships is 0.0. The mechanism stays
# in place (ScoredSentence.bigram_score, the clause-selection tiebreak) since
# it's still structurally correct and may earn a nonzero weight once
# retrieval quality improves (see the spec's "Out of scope" section on
# BM25's recall ceiling); re-run the sweep if that changes.
_BIGRAM_BONUS = 0.0


@dataclass(frozen=True)
class ScoredSentence:
    score: float
    # The bigram-bonus component alone, separate from the total -- used as
    # a tiebreak signal by generation.py's clause selection when two
    # candidates land on the same total score.
    bigram_score: float
    # Count of *content-word* (non-stopword) query tokens matched -- the
    # meaningful-evidence gate. A sentence with content_overlap == 0 only
    # shares stopwords with the query and must not count as evidence.
    content_overlap: int
    matched_tokens: list[str]


def score_sentence(query_tokens: list[str], sentence_tokens: list[str], language: str) -> ScoredSentence:
    """Weighted lexical relevance of one candidate sentence against a query.
    `language` is required (never assumed/defaulted to English) since the
    stopword set is per-language; an unrecognized language falls back to no
    stopword down-weighting rather than raising, since scoring must never be
    the reason a hot-path request fails.

    Callers are expected to tokenize both query and sentence once (via
    text_utils.tokenize_ordered) and reuse the result across calls -- this
    function does no caching of its own."""
    stopwords = _STOPWORDS.get(language, frozenset())
    sentence_token_set = set(sentence_tokens)

    matched: list[str] = []
    seen: set[str] = set()
    base_score = 0.0
    content_overlap = 0
    for token in query_tokens:
        if token in seen or token not in sentence_token_set:
            continue
        seen.add(token)
        matched.append(token)
        if token in stopwords:
            base_score += _STOPWORD_WEIGHT
        else:
            base_score += _CONTENT_WEIGHT
            content_overlap += 1

    bigram_score = 0.0
    if len(query_tokens) > 1 and len(sentence_tokens) > 1:
        query_bigrams = {(query_tokens[i], query_tokens[i + 1]) for i in range(len(query_tokens) - 1)}
        sentence_bigrams = {(sentence_tokens[i], sentence_tokens[i + 1]) for i in range(len(sentence_tokens) - 1)}
        bigram_score = _BIGRAM_BONUS * len(query_bigrams & sentence_bigrams)

    return ScoredSentence(
        score=base_score + bigram_score,
        bigram_score=bigram_score,
        content_overlap=content_overlap,
        matched_tokens=matched,
    )
