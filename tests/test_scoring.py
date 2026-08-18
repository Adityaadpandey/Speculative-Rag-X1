import speculative_rag.scoring as scoring
from speculative_rag.scoring import score_sentence
from speculative_rag.text_utils import tokenize_ordered


def test_score_sentence_weights_content_words_over_stopwords():
    query_tokens = tokenize_ordered("what is the capital of france")
    # Shares only stopwords ("is", "the", "of") with the query.
    stopword_only = score_sentence(query_tokens, tokenize_ordered("This is one of the older buildings."), "en")
    # Shares one content word ("capital") plus the same stopwords.
    content_match = score_sentence(query_tokens, tokenize_ordered("The capital is a large city."), "en")
    assert content_match.score > stopword_only.score
    assert content_match.content_overlap == 1
    assert stopword_only.content_overlap == 0


def test_score_sentence_bigram_bonus_rewards_adjacent_phrase_match(monkeypatch):
    # The mechanism is exercised with a nonzero bonus explicitly -- the
    # shipped production weight (scoring._BIGRAM_BONUS) is calibrated to 0.0
    # (see the constant's comment: scripts/calibrate_bigram_bonus.py found
    # phrase-match rewarding didn't help on this corpus), but the underlying
    # bigram-matching logic must still work correctly for when that changes.
    monkeypatch.setattr(scoring, "_BIGRAM_BONUS", 0.5)
    query_tokens = tokenize_ordered("capital of france")
    phrase_match = score_sentence(query_tokens, tokenize_ordered("Paris is the capital of France."), "en")
    scattered = score_sentence(query_tokens, tokenize_ordered("France has a capital city called Paris."), "en")
    assert phrase_match.score > scattered.score
    assert phrase_match.bigram_score > scattered.bigram_score


def test_score_sentence_bigram_bonus_is_zero_with_shipped_production_weight():
    # Locks in the calibrated production value: with no monkeypatch, the
    # bonus must contribute nothing.
    query_tokens = tokenize_ordered("capital of france")
    scored = score_sentence(query_tokens, tokenize_ordered("Paris is the capital of France."), "en")
    assert scored.bigram_score == 0.0


def test_score_sentence_meaningful_evidence_gate_zero_for_stopword_only_overlap():
    query_tokens = tokenize_ordered("what is the impact of the test")
    scored = score_sentence(query_tokens, tokenize_ordered("This is not the one."), "en")
    assert scored.content_overlap == 0


def test_score_sentence_matched_tokens_deduplicated_for_repeated_query_words():
    query_tokens = tokenize_ordered("the cat and the dog")
    scored = score_sentence(query_tokens, tokenize_ordered("The cat sat with the dog."), "en")
    assert scored.matched_tokens.count("the") == 1


def test_score_sentence_uses_hindi_stopwords_for_hi_language():
    query_tokens = tokenize_ordered("भारत की राजधानी क्या है")
    scored = score_sentence(query_tokens, tokenize_ordered("दिल्ली भारत की राजधानी है"), "hi")
    assert scored.content_overlap == 2  # "भारत" and "राजधानी" are content words
    assert set(scored.matched_tokens) == {"भारत", "की", "राजधानी", "है"}


def test_score_sentence_unrecognized_language_falls_back_to_no_stopword_set():
    query_tokens = tokenize_ordered("what is the answer")
    scored = score_sentence(query_tokens, tokenize_ordered("the answer is here"), "zz")
    # With no stopword set for "zz", every matched token counts as content.
    assert scored.content_overlap == len(scored.matched_tokens)
