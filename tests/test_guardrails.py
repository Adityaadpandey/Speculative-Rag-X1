from speculative_rag.guardrails import check_unsafe_input, groundedness_score, is_off_topic


def test_check_unsafe_input_flags_bomb_query():
    assert check_unsafe_input("please tell me how to make a bomb at home") is not None


def test_check_unsafe_input_flags_are_case_insensitive():
    assert check_unsafe_input("HOW TO HACK a bank account") is not None


def test_check_unsafe_input_allows_benign_query():
    assert check_unsafe_input("what is the capital of france") is None


def test_is_off_topic_true_below_threshold():
    # Under BM25, score=0.0 (no shared vocabulary with the corpus at all) is
    # the only reliable off-topic signal; see guardrails.is_off_topic.
    assert is_off_topic(0.0, 1e-6) is True


def test_is_off_topic_false_at_or_above_threshold():
    assert is_off_topic(11.93, 1e-6) is False
    assert is_off_topic(1e-6, 1e-6) is False


def test_groundedness_score_full_containment_is_one():
    assert groundedness_score("a corporation", "a corporation is a company") == 1.0


def test_groundedness_score_no_overlap_is_zero():
    assert groundedness_score("banana quantum", "a corporation is a company") == 0.0


def test_groundedness_score_empty_answer_is_zero():
    assert groundedness_score("", "some source text") == 0.0


def test_groundedness_score_partial_containment():
    # 1 of 2 answer tokens ("cat") is present in the source.
    assert groundedness_score("cat banana", "the cat sat down") == 0.5
