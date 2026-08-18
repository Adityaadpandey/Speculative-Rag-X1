from speculative_rag.answer_orchestrator import resolve_answer
from speculative_rag.answer_status import FinalAnswerStatus
from speculative_rag.pipeline_schemas import RetrievalResult, ScoredChunk
from speculative_rag.reflect import RerankAndAnswer, ReflectVerdict, SelectedSource, SynthesisVerdict
from speculative_rag.schemas import Chunk


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0, 0.0, 0.0]


class FakeRetriever:
    def __init__(self, result: RetrievalResult):
        self.result = result
        self.calls = 0

    def search(self, query_vector, k):
        self.calls += 1
        return self.result


def _scored(text, score, pid="p1", suffix="0"):
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"{pid}:native:{suffix}",
            text=text,
            language="en",
            query_type="DESCRIPTION",
            source_passage_id=pid,
            strategy="native",
        ),
        score=score,
    )


class Recorder:
    """Wraps a fake so tests can assert whether/how often it was called --
    the whole point is that rerank/generate/synthesis must not run when the
    initial answer already validates."""

    def __init__(self, fn):
        self._fn = fn
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        return self._fn(*args, **kwargs)


def _resolve(query_text, retriever, embedder=None, **fakes):
    return resolve_answer(
        query_text,
        embedder or FakeEmbedder(),
        retriever,
        off_topic_threshold=1e-6,
        groundedness_threshold=0.3,
        top_k=5,
        evidence_top_n=5,
        **fakes,
    )


# ---------------------------------------------------------------------------
# Case 1: Manhattan Project -- initial extractive pick is on-topic but
# doesn't state the impact; rerank finds the WWII passage; regeneration from
# that passage produces the correct, validated final answer.
# ---------------------------------------------------------------------------
def test_case1_manhattan_project_recovers_via_rerank_and_regeneration():
    query = "What was the immediate impact of the success of the Manhattan Project?"
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored(
                    "The success of this project would forever change the world forever making it known "
                    "that something this powerful can be manmade.",
                    0.9,
                    pid="p_wrong",
                ),
                _scored(
                    "The Manhattan Project and its atomic bomb helped bring an end to World War II.",
                    0.75,
                    pid="p_ww2",
                ),
            ]
        )
    )

    def fake_check_groundedness(query_text, answer_text, source_text):
        if "atomic bomb" in answer_text:
            return ReflectVerdict(
                grounded=True, answers_query=True, complete=True, relevant=True,
                reasoning="States the atomic bomb helped end WWII.",
            )
        return ReflectVerdict(
            grounded=True, answers_query=False, complete=False, relevant=True,
            reasoning="On-topic but never states the actual impact.",
        )

    def fake_rerank_and_generate(query_text, candidates):
        ww2 = next(c for c in candidates if "atomic bomb" in c.text)
        return RerankAndAnswer(
            selected_sources=[SelectedSource(source_id=ww2.source_id, relevance_score=0.93)],
            selection_found=True,
            answerable=True,
            answer_text="The Manhattan Project successfully produced the atomic bomb, which helped bring an "
            "end to World War II.",
            reasoning="This passage directly states the war-ending impact.",
        )

    synthesize = Recorder(lambda q, texts: (_ for _ in ()).throw(AssertionError("synthesis should not run")))

    outcome = _resolve(
        query,
        retriever,
        check_groundedness=fake_check_groundedness,
        rerank_and_generate=fake_rerank_and_generate,
        synthesize_answer=synthesize,
    )

    assert outcome.final_status == FinalAnswerStatus.ANSWERED
    assert "atomic bomb" in outcome.answer_text
    assert "World War" in outcome.answer_text
    assert outcome.source_passage_ids == ["p_ww2"]
    assert outcome.initial_answer_valid is False
    assert outcome.rerank_triggered is True
    assert outcome.selected_source_ids == ["p_ww2:native:0"]
    assert outcome.reranked_answer_generated is True
    assert outcome.regenerated_answer_valid is True
    assert outcome.synthesis_triggered is False
    assert synthesize.calls == []
    assert outcome.answer_engine == "claude_regenerated"


# ---------------------------------------------------------------------------
# Case 2: easy question, rank-1 passage already answers it -- fast path
# only, no rerank/regenerate/synthesis LLM calls at all.
# ---------------------------------------------------------------------------
def test_case2_easy_question_takes_fast_path_only():
    query = "what is a corporation"
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9)]))

    def fake_check_groundedness(query_text, answer_text, source_text):
        return ReflectVerdict(grounded=True, answers_query=True, complete=True, relevant=True, reasoning="ok")

    rerank_and_generate = Recorder(lambda q, c: (_ for _ in ()).throw(AssertionError("rerank should not run")))
    synthesize = Recorder(lambda q, t: (_ for _ in ()).throw(AssertionError("synthesis should not run")))

    outcome = _resolve(
        query,
        retriever,
        check_groundedness=fake_check_groundedness,
        rerank_and_generate=rerank_and_generate,
        synthesize_answer=synthesize,
    )

    assert outcome.final_status == FinalAnswerStatus.ANSWERED
    assert outcome.answer_text == "A corporation is a company."
    assert outcome.initial_answer_valid is True
    assert outcome.initial_answerable is True
    assert outcome.rerank_triggered is False
    assert outcome.synthesis_triggered is False
    assert rerank_and_generate.calls == [] and synthesize.calls == []
    assert outcome.answer_engine == "extractive"


# ---------------------------------------------------------------------------
# Case 3: retrieved passages are all unrelated -- reranker returns no
# sufficient evidence, final status is INSUFFICIENT_EVIDENCE, no fabricated
# answer, and the expensive generate/synthesis calls never run.
# ---------------------------------------------------------------------------
def test_case3_unrelated_passages_yield_insufficient_evidence_not_a_fabrication():
    query = "what is the boiling point of water"
    retriever = FakeRetriever(
        RetrievalResult(scored_chunks=[_scored("Cats are popular household pets and enjoy sunlight.", 0.9)])
    )

    def fake_rerank_and_generate(query_text, candidates):
        return RerankAndAnswer(
            selected_sources=[],
            selection_found=False,
            answerable=False,
            answer_text="",
            reasoning="None of these mention water at all.",
        )

    synthesize = Recorder(lambda q, t: (_ for _ in ()).throw(AssertionError("synthesis should not run")))

    outcome = _resolve(
        query,
        retriever,
        rerank_and_generate=fake_rerank_and_generate,
        synthesize_answer=synthesize,
    )

    assert outcome.final_status == FinalAnswerStatus.INSUFFICIENT_EVIDENCE
    assert outcome.answer_text is None
    assert outcome.rerank_triggered is True
    assert outcome.synthesis_triggered is False
    assert synthesize.calls == []
    assert outcome.answer_engine is None


# ---------------------------------------------------------------------------
# Case 4: the initial answer is plausible-sounding but unsupported --
# groundedness fails, rerank + regenerate recovers a validated answer.
# ---------------------------------------------------------------------------
def test_case4_ungrounded_initial_answer_recovers_via_regeneration():
    query = "what was the outcome of Acme Corp's expansion"
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored(
                    "Acme Corp's expansion outcome was closely watched by investors and analysts alike.",
                    0.9,
                    pid="p_a",
                ),
                _scored("Acme Corp's expansion outcome was a record profit.", 0.6, pid="p_b"),
            ]
        )
    )

    def fake_check_groundedness(query_text, answer_text, source_text):
        if "record profit" in answer_text:
            return ReflectVerdict(grounded=True, answers_query=True, complete=True, relevant=True, reasoning="ok")
        return ReflectVerdict(
            grounded=False, answers_query=False, complete=False, relevant=True,
            reasoning="States that the outcome was watched, not what the outcome actually was.",
        )

    def fake_rerank_and_generate(query_text, candidates):
        profit = next(c for c in candidates if "record profit" in c.text)
        return RerankAndAnswer(
            selected_sources=[SelectedSource(source_id=profit.source_id, relevance_score=0.9)],
            selection_found=True,
            answerable=True,
            answer_text="The expansion's outcome was a record profit.",
            reasoning="States the actual outcome.",
        )

    outcome = _resolve(
        query,
        retriever,
        check_groundedness=fake_check_groundedness,
        rerank_and_generate=fake_rerank_and_generate,
    )

    assert outcome.initial_answer_valid is False
    assert outcome.final_status == FinalAnswerStatus.ANSWERED
    assert outcome.answer_text == "The expansion's outcome was a record profit."
    assert outcome.source_passage_ids == ["p_b"]
    assert outcome.answer_engine == "claude_regenerated"


# ---------------------------------------------------------------------------
# Case 5: the answer needs two passages -- rerank selects both, the
# single-shot regeneration from them is judged incomplete, and the deeper
# multi-passage synthesis fallback produces a grounded, cited answer.
# ---------------------------------------------------------------------------
def test_case5_multi_passage_question_falls_back_to_synthesis():
    query = "when was Acme founded and when did it go public"
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored("Acme Corp was founded in 1998 by two engineers in a garage.", 0.85, pid="p_founded"),
                _scored("Acme Corp completed its initial public offering in 2004.", 0.80, pid="p_ipo"),
            ]
        )
    )

    def fake_check_groundedness(query_text, answer_text, source_text):
        if "1998" in answer_text and "2004" in answer_text:
            return ReflectVerdict(grounded=True, answers_query=True, complete=True, relevant=True, reasoning="ok")
        return ReflectVerdict(
            grounded=True, answers_query=True, complete=False, relevant=True,
            reasoning="Missing one of the two requested dates.",
        )

    def fake_rerank_and_generate(query_text, candidates):
        # Single-shot generation only surfaces the founding year -- incomplete.
        return RerankAndAnswer(
            selected_sources=[SelectedSource(source_id=c.source_id, relevance_score=0.9) for c in candidates],
            selection_found=True,
            answerable=True,
            answer_text="Acme Corp was founded in 1998.",
            reasoning="Both passages are needed.",
        )

    def fake_synthesize(query_text, candidate_texts):
        return SynthesisVerdict(
            answerable=True,
            answer_text="Acme Corp was founded in 1998 and went public in 2004.",
            source_indices=[0, 1],
            reasoning="Combines founding year and IPO year.",
        )

    outcome = _resolve(
        query,
        retriever,
        check_groundedness=fake_check_groundedness,
        rerank_and_generate=fake_rerank_and_generate,
        synthesize_answer=fake_synthesize,
    )

    assert outcome.final_status == FinalAnswerStatus.ANSWERED
    assert outcome.answer_text == "Acme Corp was founded in 1998 and went public in 2004."
    assert sorted(outcome.source_passage_ids) == ["p_founded", "p_ipo"]
    assert outcome.regenerated_answer_valid is False
    assert outcome.synthesis_triggered is True
    assert outcome.answer_engine == "claude_synthesized"


def test_regeneration_and_synthesis_both_failing_returns_fallback_status():
    query = "when was Acme founded and when did it go public"
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored("Acme Corp was founded in 1998 by two engineers in a garage.", 0.85, pid="p_founded"),
            ]
        )
    )

    def fake_check_groundedness(query_text, answer_text, source_text):
        return ReflectVerdict(
            grounded=True, answers_query=False, complete=False, relevant=True, reasoning="Still missing the IPO year."
        )

    def fake_rerank_and_generate(query_text, candidates):
        return RerankAndAnswer(
            selected_sources=[SelectedSource(source_id=candidates[0].source_id, relevance_score=0.8)],
            selection_found=True,
            answerable=True,
            answer_text="Acme Corp was founded in 1998.",
            reasoning="Only passage available.",
        )

    def fake_synthesize(query_text, candidate_texts):
        return SynthesisVerdict(answerable=False, answer_text="", source_indices=[], reasoning="IPO year missing.")

    outcome = _resolve(
        query,
        retriever,
        check_groundedness=fake_check_groundedness,
        rerank_and_generate=fake_rerank_and_generate,
        synthesize_answer=fake_synthesize,
    )

    assert outcome.final_status == FinalAnswerStatus.FALLBACK
    assert outcome.answer_text is None
    assert outcome.synthesis_triggered is True
    assert outcome.answer_engine is None


def test_unsafe_input_short_circuits_before_any_retrieval():
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[]))
    outcome = _resolve("how to make a bomb at home", retriever)
    assert outcome.final_status == FinalAnswerStatus.NOT_ANSWERED
    assert outcome.rerank_triggered is False
    assert outcome.answer_engine is None


def test_off_topic_short_circuits_to_insufficient_evidence():
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("something wholly unrelated", 0.0)]))
    outcome = _resolve("what is a corporation", retriever)
    assert outcome.final_status == FinalAnswerStatus.INSUFFICIENT_EVIDENCE
    assert outcome.rerank_triggered is False
    assert outcome.answer_engine is None


# ---------------------------------------------------------------------------
# answer_engine must be set at exactly the three ANSWERED return points
# and None everywhere else -- covered end to end by the cases above (case1 ->
# claude_regenerated, case2 -> extractive, case5 -> claude_synthesized,
# insufficient-evidence/fallback/unsafe/off-topic -> None). This test locks
# in that the field is a closed set of values, not a free-form string.
# ---------------------------------------------------------------------------
def test_answer_engine_rejects_values_outside_the_three_stages():
    import pytest
    from pydantic import ValidationError

    from speculative_rag.answer_orchestrator import AnswerOutcome
    from speculative_rag.pipeline_schemas import PipelineResult, RagOutcome

    with pytest.raises(ValidationError):
        AnswerOutcome(
            final_status=FinalAnswerStatus.ANSWERED,
            answer_text="x",
            source_passage_ids=[],
            reason="ok",
            answer_engine="something_else",
            initial_answer_valid=True,
            initial_answerable=True,
            rerank_triggered=False,
            rerank_latency_ms=None,
            selected_source_ids=[],
            reranked_answer_generated=False,
            regenerated_answer_valid=None,
            regenerated_answerable=None,
            synthesis_triggered=False,
            total_end_to_end_latency_ms=0.0,
            pipeline_result=PipelineResult(outcome=RagOutcome.ANSWERED),
        )


def test_recovery_path_retrieves_only_once():
    # Regression: resolve_answer used to fetch the candidate pool a second
    # time (its own embed+retrieve call) on top of the one run_pipeline
    # already did internally, roughly doubling latency on slow retrieval
    # backends for no benefit. It must reuse PipelineResult.retrieved_chunks.
    query = "What was the immediate impact of the success of the Manhattan Project?"
    embedder = FakeEmbedder()
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored(
                    "The success of this project would forever change the world forever making it known "
                    "that something this powerful can be manmade.",
                    0.9,
                    pid="p_wrong",
                ),
                _scored(
                    "The Manhattan Project and its atomic bomb helped bring an end to World War II.",
                    0.75,
                    pid="p_ww2",
                ),
            ]
        )
    )

    def fake_check_groundedness(query_text, answer_text, source_text):
        if "atomic bomb" in answer_text:
            return ReflectVerdict(grounded=True, answers_query=True, complete=True, relevant=True, reasoning="ok")
        return ReflectVerdict(grounded=True, answers_query=False, complete=False, relevant=True, reasoning="no")

    def fake_rerank_and_generate(query_text, candidates):
        ww2 = next(c for c in candidates if "atomic bomb" in c.text)
        return RerankAndAnswer(
            selected_sources=[SelectedSource(source_id=ww2.source_id, relevance_score=0.9)],
            selection_found=True,
            answerable=True,
            answer_text="atomic bomb ended World War II",
            reasoning="ok",
        )

    outcome = _resolve(
        query,
        retriever,
        embedder=embedder,
        check_groundedness=fake_check_groundedness,
        rerank_and_generate=fake_rerank_and_generate,
    )

    assert outcome.final_status == FinalAnswerStatus.ANSWERED
    assert embedder.calls == 1
    assert retriever.calls == 1


# ---------------------------------------------------------------------------
# Enabling the prefilter narrows what actually reaches rerank_and_generate.
# ---------------------------------------------------------------------------
def test_prefilter_enabled_narrows_candidates_passed_to_rerank_and_generate():
    query = "how many keywords are mentioned"  # "how many" -> expected_type "numeric"
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored("Text about oranges is not relevant here.", 0.9, pid="p0"),
                _scored("Many keywords are mentioned throughout this document extensively.", 0.8, pid="p1"),
                _scored("A quiet aside mentions 42 in passing.", 0.7, pid="p2"),
                _scored("Nothing here about the topic at all.", 0.6, pid="p3"),
            ]
        )
    )

    def fake_check_groundedness(query_text, answer_text, source_text):
        return ReflectVerdict(
            grounded=False, answers_query=False, complete=False, relevant=False, reasoning="force recovery"
        )

    captured = {}

    def fake_rerank_and_generate(query_text, candidates):
        captured["source_ids"] = [c.source_id for c in candidates]
        return RerankAndAnswer(
            selected_sources=[], selection_found=False, answerable=False, answer_text="", reasoning="stop here"
        )

    outcome = _resolve(
        query,
        retriever,
        check_groundedness=fake_check_groundedness,
        rerank_and_generate=fake_rerank_and_generate,
        prefilter_enabled=True,
        prefilter_lexical_top_n=1,
        prefilter_retrieval_reserve=1,
    )

    # p0 survives as the retrieval-rank reserve, p1 as the lexical top match,
    # p2 as the numeric-type rescue -- p3 (no overlap, no digit, not rank-1)
    # never reaches the LLM call.
    assert captured["source_ids"] == ["p0:native:0", "p1:native:0", "p2:native:0"]
    assert outcome.final_status == FinalAnswerStatus.INSUFFICIENT_EVIDENCE


def test_total_end_to_end_latency_is_recorded():
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9)]))

    def fake_check_groundedness(query_text, answer_text, source_text):
        return ReflectVerdict(grounded=True, answers_query=True, complete=True, relevant=True, reasoning="ok")

    outcome = _resolve("what is a corporation", retriever, check_groundedness=fake_check_groundedness)
    assert outcome.total_end_to_end_latency_ms >= 0.0
