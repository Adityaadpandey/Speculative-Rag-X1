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


def test_select_span_prefers_type_matching_sentence_over_higher_overlap_topical_one():
    # Regression case from results/answer_quality_en_native.md: a numeric
    # query's top-overlap sentence merely discusses the topic without a
    # number ("good credit"), while a lower-overlap sentence actually holds
    # the number ("660"). Raw overlap alone picks the wrong one; the
    # expected-answer-type filter should surface the numeric sentence.
    passage = (
        "You will need to have good credit and provide proof of income for the card. "
        "A score of 660 or better is required."
    )
    span = select_answer_span("what credit score do you need for the card", passage)
    assert span == "A score of 660 or better is required."


def test_select_span_breaks_overlap_tie_toward_later_sentence():
    # Real production regression: the intro sentence restates the passage's
    # own subject ("Manhattan Project") and ties on raw overlap with the
    # sentence that actually answers the query, since "impact" never appears
    # verbatim anywhere -- lexical overlap alone can't tell them apart. When
    # tied, prefer the later sentence: definition-then-consequence is the
    # dominant pattern for "what was the impact/result of X" queries.
    passage = (
        "Essay on The Manhattan Project - The Manhattan Project The Manhattan "
        "Project was to see if making an atomic bomb possible. "
        "The success of this project would forever change the world forever "
        "making it known that something this powerful can be manmade."
    )
    span = select_answer_span(
        "what was the immediate impact of the success of the manhattan project?", passage
    )
    assert span.startswith("The success of this project")


def test_select_span_falls_back_to_overlap_when_no_candidate_satisfies_type():
    # Numeric query, but no sentence in the passage contains a digit at all —
    # the type filter must not eliminate every candidate; fall back to the
    # plain overlap-ranked choice instead of returning None.
    passage = "The card requires excellent credit. Approval also depends on your income."
    span = select_answer_span("what credit score do you need for the card", passage)
    assert span == "The card requires excellent credit."


def test_select_span_strips_duplicate_tail_corruption():
    # Real corpus artifact: some passages contain their own opening text
    # repeated later in the same passage, glued on with no separating
    # whitespace and missing its first character (e.g. a live "en_native"
    # passage whose text ended "...Germany.Alphabetical list of countries in
    # Europe.he most visited travel destination..." -- its own opening
    # sentence reappearing with the leading "T" dropped). Without stripping
    # it, the extracted span would be the sentence stitched to a mangled
    # copy of itself instead of a clean, single answer.
    original = "The capital of Ruritania is Strelsauville, a scenic city on the northern bank of the river Endar. "
    passage = original + original[1:]
    span = select_answer_span("what is the capital of Ruritania", passage)
    assert span == original.strip()
    assert span.count("Strelsauville") == 1


def test_select_span_splits_sentence_glued_to_next_by_missing_space():
    # Companion corpus artifact: real sentence boundaries sometimes lose
    # their separating space (e.g. "...Germany.Alphabetical list...").
    # split_sentences must still isolate the query-relevant sentence rather
    # than returning it merged with an unrelated trailing clause.
    passage = "The capital of Ruritania is Strelsau.Alphabetical list of Zendan cities follows."
    span = select_answer_span("what is the capital of Ruritania", passage)
    assert span == "The capital of Ruritania is Strelsau."
