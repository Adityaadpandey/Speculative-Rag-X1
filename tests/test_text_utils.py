from speculative_rag.text_utils import tokenize, tokenize_ordered


def test_tokenize_lowercases_and_splits_on_word_boundaries():
    assert tokenize("A Corporation, is!") == {"a", "corporation", "is"}


def test_tokenize_empty_string_is_empty_set():
    assert tokenize("") == set()


def test_tokenize_handles_non_ascii_scripts():
    # \w with re.UNICODE keeps Devanagari word characters as tokens.
    tokens = tokenize("दिल्ली भारत")
    assert tokens == {"दिल्ली", "भारत"}


def test_tokenize_handles_tamil_combining_marks():
    # Tamil vowel signs/virama are combining marks (category Mn/Mc) that
    # must stay attached to their base letter, not become separate tokens.
    tokens = tokenize("தமிழ் மொழி")
    assert tokens == {"தமிழ்", "மொழி"}


def test_tokenize_ordered_preserves_order_and_duplicates():
    assert tokenize_ordered("A Corporation is a corporation!") == ["a", "corporation", "is", "a", "corporation"]


def test_tokenize_ordered_empty_string_is_empty_list():
    assert tokenize_ordered("") == []


def test_tokenize_ordered_handles_devanagari():
    assert tokenize_ordered("दिल्ली भारत की दिल्ली") == ["दिल्ली", "भारत", "की", "दिल्ली"]
