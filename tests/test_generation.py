from speculative_rag.generation import select_answer_span


def test_select_span_picks_sentence_overlapping_query():
    passage = "Cats are mammals. The Eiffel Tower is in Paris. Dogs bark loudly."
    span = select_answer_span("where is the Eiffel Tower", passage)
    assert "Eiffel Tower" in span
    assert "Cats" not in span


def test_select_span_single_sentence_returns_whole_passage():
    passage = "A corporation is a legal entity"
    assert select_answer_span("what is a corporation", passage) == "A corporation is a legal entity"


def test_select_span_no_overlap_returns_none_instead_of_dumping_passage():
    # This is the exact failure mode from results/answer_quality_en_native.md:
    # a topically-plausible passage with zero query-token overlap used to be
    # returned WHOLE, then scored groundedness=1.0 trivially (the "answer"
    # *is* the source by construction) and marked ANSWERED. It must now
    # signal "no answer here" instead of masquerading as one.
    passage = "Cats are mammals. Dogs bark loudly."
    span = select_answer_span("quantum chromodynamics theory", passage)
    assert span is None


def test_select_span_picks_only_relevant_sentence_among_many_irrelevant():
    # Adversarial case: passage contains the answer plus 10 irrelevant
    # sentences -- the answer should only contain the relevant part, not a
    # dump of everything.
    passage = (
        "The weather today is sunny. Traffic was light this morning. "
        "Coffee prices rose last quarter. "
        "The immediate impact of the test was a bright flash visible for miles. "
        "Birds migrate south in winter. The stock market closed higher. "
        "Local schools are on break this week. Many tourists visited the museum. "
        "The bridge was repainted recently. A new cafe opened downtown. "
        "Rainfall was below average this year."
    )
    span = select_answer_span("what was the immediate impact of the test", passage)
    assert span == "The immediate impact of the test was a bright flash visible for miles."
