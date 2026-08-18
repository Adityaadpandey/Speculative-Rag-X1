import pytest

from speculative_rag.chunking import (
    chunk_metadata_aware,
    chunk_native,
    chunk_sentence_aware,
    chunk_sliding_window,
    split_sentences,
)
from speculative_rag.schemas import Passage


def _passage(text):
    return Passage(passage_id="p1", text=text, language="en", query_type="DESCRIPTION")


def test_chunk_native_returns_single_chunk():
    chunks = chunk_native(_passage("A short passage."))
    assert len(chunks) == 1
    assert chunks[0].text == "A short passage."
    assert chunks[0].strategy == "native"
    assert chunks[0].source_passage_id == "p1"


def test_chunk_sliding_window_short_passage_equals_native():
    short = _passage(" ".join(["word"] * 10))
    chunks = chunk_sliding_window(short, window_words=40, overlap_words=10)
    assert len(chunks) == 1
    assert chunks[0].text == short.text


def test_chunk_sliding_window_long_passage_splits_with_overlap():
    long_text = " ".join(f"w{i}" for i in range(100))
    long_passage = _passage(long_text)
    chunks = chunk_sliding_window(long_passage, window_words=40, overlap_words=10)

    assert len(chunks) > 1
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-10:] == second_words[:10]  # overlap region matches
    assert all(c.source_passage_id == "p1" for c in chunks)


def test_split_sentences_handles_devanagari_danda():
    sentences = split_sentences("यह पहला वाक्य है। यह दूसरा वाक्य है।")
    assert len(sentences) == 2


def test_split_sentences_splits_on_missing_space_before_capital():
    # Corpus artifact: some passages lose the space at a sentence boundary
    # (e.g. "...Germany.Alphabetical list..."). Without also splitting on
    # punctuation directly followed by an uppercase letter, the two
    # sentences stay glued together and an irrelevant trailing clause leaks
    # into whatever span gets extracted from the first one.
    sentences = split_sentences("First sentence ends here.Second sentence starts here.")
    assert sentences == ["First sentence ends here.", "Second sentence starts here."]


def test_split_sentences_missing_space_is_noop_for_devanagari():
    # Devanagari/Tamil have no letter case, so the missing-space-before-
    # capital heuristic must never fire there -- it would have nothing
    # meaningful to key off and shouldn't introduce spurious splits.
    sentences = split_sentences("यह पहला वाक्य है।यह दूसरा वाक्य है।")
    assert len(sentences) == 1


def test_chunk_sentence_aware_short_passage_equals_native():
    short = _passage("One short sentence.")
    chunks = chunk_sentence_aware(short, max_words=40, overlap_sentences=1)
    assert len(chunks) == 1
    assert chunks[0].text == "One short sentence."


def test_chunk_sentence_aware_long_passage_splits_on_boundaries():
    sentences = [f"Sentence number {i} has several words in it." for i in range(10)]
    long_passage = _passage(" ".join(sentences))
    chunks = chunk_sentence_aware(long_passage, max_words=20, overlap_sentences=1)

    assert len(chunks) > 1
    for c in chunks:
        assert c.text.strip().endswith(".")


def test_chunk_sliding_window_rejects_overlap_not_smaller_than_window():
    with pytest.raises(ValueError):
        chunk_sliding_window(_passage("some text"), window_words=20, overlap_words=20)


def test_split_sentences_protects_single_abbreviation():
    assert split_sentences("Dr. Smith arrived.") == ["Dr. Smith arrived."]


def test_split_sentences_protects_adjacent_abbreviations():
    assert split_sentences("Mr. and Mrs. Smith left.") == ["Mr. and Mrs. Smith left."]


def test_split_sentences_protects_lowercase_abbreviations():
    assert split_sentences("e.g. this is common.") == ["e.g. this is common."]


def test_split_sentences_protects_multi_period_abbreviation():
    assert split_sentences("The U.S. government announced today.") == ["The U.S. government announced today."]


def test_split_sentences_protects_no_abbreviation():
    assert split_sentences("See No. 5 for details.") == ["See No. 5 for details."]


def test_split_sentences_decimal_numbers_already_safe():
    # Regression lock-in: the whitespace branch requires \s+ after the
    # punctuation and the uppercase-lookahead branch requires an uppercase
    # Latin letter immediately after -- neither is satisfied by a digit, so
    # this was already safe before the abbreviation fix and must stay safe.
    assert split_sentences("The score was 3.14 exactly.") == ["The score was 3.14 exactly."]


def test_split_sentences_protects_st_abbreviation():
    assert split_sentences("St. Louis is a city.") == ["St. Louis is a city."]


def test_split_sentences_still_splits_real_boundary_after_abbreviation():
    # The protect/split/restore fix must not swallow a genuine sentence
    # boundary that happens to follow an abbreviation.
    assert split_sentences("Dr. Smith arrived. He was late.") == ["Dr. Smith arrived.", "He was late."]


def test_chunk_metadata_aware_matches_native_boundaries_but_tags_strategy():
    p = _passage("A short passage.")
    native = chunk_native(p)[0]
    meta = chunk_metadata_aware(p)[0]

    assert meta.text == native.text
    assert meta.strategy == "metadata"
    assert meta.language == "en"
    assert meta.query_type == "DESCRIPTION"
