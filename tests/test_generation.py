from speculative_rag.generation import (
    complete_answer_from_evidence,
    prefilter_rerank_candidates,
    select_answer_span,
)
from speculative_rag.pipeline_schemas import ScoredChunk
from speculative_rag.schemas import Chunk


def _scored(text, score, pid="p1", suffix="0", language="en"):
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"{pid}:native:{suffix}",
            text=text,
            language=language,
            query_type="DESCRIPTION",
            source_passage_id=pid,
            strategy="native",
        ),
        score=score,
    )


def test_complete_answer_from_evidence_expands_a_terse_prefix():
    answer = complete_answer_from_evidence(
        "India is a peninsula",
        ["India is a peninsula, bound by the Indian Ocean in the south and the Arabian Sea on the west."],
    )
    assert answer == "India is a peninsula, bound by the Indian Ocean in the south and the Arabian Sea on the west."


def test_complete_answer_from_evidence_keeps_non_prefix_answer():
    answer = complete_answer_from_evidence(
        "India is in South Asia",
        ["India is a peninsula, bound by the Indian Ocean in the south."],
    )
    assert answer == "India is in South Asia"


def test_select_span_picks_sentence_overlapping_query():
    passage = "Cats are mammals. The Eiffel Tower is in Paris. Dogs bark loudly."
    span = select_answer_span("where is the Eiffel Tower", passage, "en")
    assert "Eiffel Tower" in span
    assert "Cats" not in span


def test_select_span_single_sentence_returns_whole_passage():
    passage = "A corporation is a legal entity"
    assert select_answer_span("what is a corporation", passage, "en") == "A corporation is a legal entity"


def test_select_span_no_overlap_returns_none_instead_of_dumping_passage():
    # This is the exact failure mode from results/answer_quality_en_native.md:
    # a topically-plausible passage with zero query-token overlap used to be
    # returned WHOLE, then scored groundedness=1.0 trivially (the "answer"
    # *is* the source by construction) and marked ANSWERED. It must now
    # signal "no answer here" instead of masquerading as one.
    passage = "Cats are mammals. Dogs bark loudly."
    span = select_answer_span("quantum chromodynamics theory", passage, "en")
    assert span is None


def test_select_span_single_sentence_zero_overlap_returns_none():
    # Regression fix: the single-sentence fallback used to return passage_text.strip()
    # unconditionally, even with zero query overlap -- inconsistent with the
    # multi-sentence path. It must now apply the same meaningful-evidence
    # gate and return None instead of dumping the passage as if it answered.
    passage = "Cats are mammals"
    span = select_answer_span("quantum chromodynamics theory", passage, "en")
    assert span is None


def test_select_span_single_sentence_stopword_only_overlap_returns_none():
    # Sharing only stopwords ("is", "of", "the") with the query is not
    # meaningful evidence -- the content-word-only gate must reject this
    # even though a naive token-set intersection would call it a match.
    passage = "This is one of the many buildings."
    span = select_answer_span("what is the capital of France", passage, "en")
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
    span = select_answer_span("what was the immediate impact of the test", passage, "en")
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
    span = select_answer_span("what credit score do you need for the card", passage, "en")
    assert span == "A score of 660 or better is required."


def test_select_span_breaks_overlap_tie_toward_later_sentence():
    # Real production regression: the intro sentence restates the passage's
    # own subject ("Manhattan Project") and ties on weighted score with the
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
        "what was the immediate impact of the success of the manhattan project?", passage, "en"
    )
    assert span.startswith("The success of this project")


def test_select_span_falls_back_to_overlap_when_no_candidate_satisfies_type():
    # Numeric query, but no sentence in the passage contains a digit at all —
    # the type filter must not eliminate every candidate; fall back to the
    # plain overlap-ranked choice instead of returning None.
    passage = "The card requires excellent credit. Approval also depends on your income."
    span = select_answer_span("what credit score do you need for the card", passage, "en")
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
    span = select_answer_span("what is the capital of Ruritania", passage, "en")
    assert span == original.strip()
    assert span.count("Strelsauville") == 1


def test_select_span_removes_repeated_middle_sentence_with_dropped_character():
    # Real corpus artifact: a sentence is repeated later with its first
    # character dropped ("India ..." becomes "ndia ..."), glued after a
    # different sentence. The malformed copy must not become the answer.
    passage = (
        "India has seven neighbours.India is a peninsula, bound by the Indian Ocean "
        "in the south, the Arabian Sea on the west and Bay of Bengal in the east. "
        "India gets its name from the Greek word, Indus.ndia is a peninsula, bound "
        "by the Indian Ocean in the south, the Arabian Sea on the west and Bay of "
        "Bengal in the east."
    )
    span = select_answer_span("how is India a peninsula", passage, "en")
    assert span.startswith("India is a peninsula")
    assert "Indus.ndia" not in span


def test_select_span_splits_sentence_glued_to_next_by_missing_space():
    # Companion corpus artifact: real sentence boundaries sometimes lose
    # their separating space (e.g. "...Germany.Alphabetical list...").
    # split_sentences must still isolate the query-relevant sentence rather
    # than returning it merged with an unrelated trailing clause.
    passage = "The capital of Ruritania is Strelsau.Alphabetical list of Zendan cities follows."
    span = select_answer_span("what is the capital of Ruritania", passage, "en")
    assert span == "The capital of Ruritania is Strelsau."


def test_select_span_prefers_bigram_phrase_match_over_scattered_overlap(monkeypatch):
    # Two sentences share the same content-word overlap count, but one
    # has the query's words adjacent to each other ("capital of France") and
    # the other has them scattered -- with a nonzero bigram bonus, that
    # should break the tie toward the phrase match. Monkeypatched to a
    # nonzero weight because the shipped production value is calibrated to
    # 0.0 (see scoring._BIGRAM_BONUS's comment) -- this test exercises the
    # mechanism itself, not the current production tuning.
    import speculative_rag.scoring as scoring

    monkeypatch.setattr(scoring, "_BIGRAM_BONUS", 0.5)
    passage = "Paris is the capital of France. France has a capital city called Paris."
    span = select_answer_span("capital of france", passage, "en")
    assert span == "Paris is the capital of France."


def test_select_span_trims_to_clause_satisfying_expected_type():
    # A numeric query whose winning sentence has multiple comma-delimited
    # clauses, only one of which actually contains a number -- plain
    # type-matching at the sentence level would return the whole sentence;
    # the clause trim should narrow it to just the clause with the number.
    passage = "The festival lasted three days, took place in the city park, drew a crowd of 5000 people."
    span = select_answer_span("how many people attended the festival", passage, "en")
    assert span == "drew a crowd of 5000 people."


def test_select_span_returns_bare_directional_location_item():
    # Real corpus sentence shape (en_native, Karnataka passage): a shared
    # subject+verb followed by a comma list of "NP to the DIRECTION" items.
    # For "where is Goa", returning the lead clause would incorrectly make
    # Karnataka look like the answer. Restore the omitted subject/verb so the
    # location-bearing list item becomes a natural standalone answer.
    passage = (
        "Karnataka is bordered by the Arabian Sea to the west, Goa to the north west, "
        "Maharashtra to the north, and Kerala to the south west."
    )
    span = select_answer_span("where is goa", passage, "en")
    assert span == "Goa is to the north west of Karnataka."


def test_select_span_rewrites_final_directional_location_fragment():
    passage = "Karnataka is bordered by the Arabian Sea to the west, Goa to the north west."
    assert select_answer_span("where is Goa", passage, "en") == "Goa is to the north west of Karnataka."


def test_select_span_strips_leading_connective_without_corrupting_acronym():
    # The leading-connective strip must uppercase only the first cased
    # character of the remainder, not run str.capitalize() -- capitalize()
    # would lowercase the rest of the string, turning "NASA" into "Nasa".
    passage = "The rover landed safely. However, the NASA team detected a signal delay."
    span = select_answer_span("what team detected a signal delay", passage, "en")
    assert span == "The NASA team detected a signal delay."


# ---------------------------------------------------------------------------
# prefilter_rerank_candidates
# ---------------------------------------------------------------------------
def test_prefilter_keeps_union_of_lexical_top_retrieval_reserve_and_type_match():
    query = "how many keywords are mentioned"  # "how many" -> expected_type "numeric"
    c0 = _scored("Text about oranges is not relevant here.", 0.9, pid="p0")
    c1 = _scored("Many keywords are mentioned throughout this document extensively.", 0.8, pid="p1")
    c2 = _scored("A quiet aside mentions 42 in passing.", 0.7, pid="p2")
    c3 = _scored("Nothing here about the topic at all.", 0.6, pid="p3")

    survivors = prefilter_rerank_candidates(query, [c0, c1, c2, c3], lexical_top_n=1, retrieval_reserve=1)

    survivor_ids = {c.chunk.source_passage_id for c in survivors}
    # c0 survives as the retrieval-rank reserve, c1 as the lexical top match,
    # c2 as the numeric-type rescue despite near-zero lexical overlap, and c3
    # (no overlap, no digit, not rank-1) is correctly excluded.
    assert survivor_ids == {"p0", "p1", "p2"}
    assert [c.chunk.source_passage_id for c in survivors] == ["p0", "p1", "p2"]  # retrieval order preserved


def test_prefilter_skips_filtering_when_pool_has_zero_meaningful_overlap():
    # The prefilter must never be the reason a query fails: if nothing in
    # the pool shares meaningful vocabulary with the query, pass the whole
    # pool through unfiltered rather than guessing.
    query = "quantum chromodynamics theory"
    candidates = [
        _scored("Cats are popular household pets.", 0.9, pid="p0"),
        _scored("The weather today is sunny and warm.", 0.8, pid="p1"),
    ]
    survivors = prefilter_rerank_candidates(query, candidates, lexical_top_n=1, retrieval_reserve=1)
    assert survivors == candidates


def test_prefilter_empty_candidates_returns_empty():
    assert prefilter_rerank_candidates("anything", [], lexical_top_n=8, retrieval_reserve=5) == []
