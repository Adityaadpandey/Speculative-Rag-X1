from speculative_rag.text_utils import tokenize


def test_tokenize_lowercases_and_splits_on_word_boundaries():
    assert tokenize("A Corporation, is!") == {"a", "corporation", "is"}


def test_tokenize_empty_string_is_empty_set():
    assert tokenize("") == set()


def test_tokenize_handles_non_ascii_scripts():
    # \w with re.UNICODE keeps Devanagari word characters as tokens.
    tokens = tokenize("दिल्ली भारत")
    assert tokens == {"दिल्ली", "भारत"}
