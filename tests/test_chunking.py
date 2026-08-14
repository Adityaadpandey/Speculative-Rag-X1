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


def test_chunk_metadata_aware_matches_native_boundaries_but_tags_strategy():
    p = _passage("A short passage.")
    native = chunk_native(p)[0]
    meta = chunk_metadata_aware(p)[0]

    assert meta.text == native.text
    assert meta.strategy == "metadata"
    assert meta.language == "en"
    assert meta.query_type == "DESCRIPTION"
