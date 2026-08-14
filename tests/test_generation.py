from speculative_rag.generation import select_answer_span


def test_select_span_picks_sentence_overlapping_query():
    passage = "Cats are mammals. The Eiffel Tower is in Paris. Dogs bark loudly."
    span = select_answer_span("where is the Eiffel Tower", passage)
    assert "Eiffel Tower" in span
    assert "Cats" not in span


def test_select_span_single_sentence_returns_whole_passage():
    passage = "A corporation is a legal entity"
    assert select_answer_span("what is a corporation", passage) == "A corporation is a legal entity"


def test_select_span_no_overlap_falls_back_to_whole_passage():
    passage = "Cats are mammals. Dogs bark loudly."
    span = select_answer_span("quantum chromodynamics theory", passage)
    assert span == "Cats are mammals. Dogs bark loudly."
