from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome, RetrievalResult, ScoredChunk
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

    def search(self, query_vector, k):
        return self.result


def _scored(text, score, pid="p1"):
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"{pid}:native:0",
            text=text,
            language="en",
            query_type="DESCRIPTION",
            source_passage_id=pid,
            strategy="native",
        ),
        score=score,
    )


def _run(query, retriever, embedder=None, off_topic=1e-6, ground=0.3, top_k=5):
    return run_pipeline(
        query,
        embedder or FakeEmbedder(),
        retriever,
        off_topic_threshold=off_topic,
        groundedness_threshold=ground,
        top_k=top_k,
    )


def test_answered_path_returns_extractive_answer_and_all_stage_timings():
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9)]))
    result = _run("what is a corporation", retriever)

    assert result.outcome == RagOutcome.ANSWERED
    assert result.answer is not None
    assert "corporation" in result.answer.text
    assert result.answer.source_passage_id == "p1"
    assert result.answer.groundedness_score == 1.0
    assert result.answer.raw_sentence == "A corporation is a company."
    assert "corporation" in result.answer.matched_tokens
    assert result.top_score == 0.9
    assert [t.stage for t in result.timings] == [
        "guardrail_unsafe",
        "embed",
        "retrieve",
        "guardrail_offtopic",
        "generate",
        "guardrail_groundedness",
        "guardrail_answer_type",
    ]


def test_unsafe_input_short_circuits_before_embedding():
    embedder = FakeEmbedder()
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("irrelevant", 0.9)]))
    result = _run("how to make a bomb", retriever, embedder=embedder)

    assert result.outcome == RagOutcome.UNSAFE_INPUT_REFUSED
    assert result.answer is None
    assert result.refusal_reason is not None
    assert embedder.calls == 0  # never embedded
    assert [t.stage for t in result.timings] == ["guardrail_unsafe"]


def test_off_topic_refusal_when_top_score_below_threshold():
    # Under BM25, score=0.0 (no shared vocabulary at all) is the only
    # reliable off-topic signal -- see guardrails.is_off_topic.
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("something unrelated", 0.0)]))
    result = _run("what is a corporation", retriever, off_topic=1e-6)

    assert result.outcome == RagOutcome.OFF_TOPIC_REFUSED
    assert result.answer is None
    assert result.top_score == 0.0
    assert [t.stage for t in result.timings] == ["guardrail_unsafe", "embed", "retrieve", "guardrail_offtopic"]


def test_off_topic_refusal_when_no_results():
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[]))
    result = _run("anything", retriever)

    assert result.outcome == RagOutcome.OFF_TOPIC_REFUSED
    assert result.answer is None


def test_ungrounded_refusal_when_groundedness_below_threshold():
    # Force the branch: even a perfectly grounded (1.0) extractive answer is
    # below an impossible 1.5 threshold, exercising the UNGROUNDED exit.
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9)]))
    result = _run("what is a corporation", retriever, ground=1.5)

    assert result.outcome == RagOutcome.UNGROUNDED_REFUSED
    assert result.answer is None
    assert [t.stage for t in result.timings] == [
        "guardrail_unsafe",
        "embed",
        "retrieve",
        "guardrail_offtopic",
        "generate",
        "guardrail_groundedness",
    ]


def test_insufficient_evidence_when_extracted_span_lacks_expected_answer_type():
    # "what year" expects a numeric answer type; the highest token-overlap
    # sentence here ("second quarter") mentions no digit at all, mirroring
    # the real "what year did facebook ipo" failure logged in
    # results/answer_quality_en_native.md.
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored(
                    "Facebook had its ipo in the second quarter. "
                    "The company was founded by Mark Zuckerberg.",
                    0.9,
                )
            ]
        )
    )
    result = _run("what year did facebook ipo", retriever)

    assert result.outcome == RagOutcome.INSUFFICIENT_EVIDENCE
    assert result.answer is None
    assert result.refusal_reason is not None
    assert [t.stage for t in result.timings] == [
        "guardrail_unsafe",
        "embed",
        "retrieve",
        "guardrail_offtopic",
        "generate",
        "guardrail_groundedness",
        "guardrail_answer_type",
    ]


def test_insufficient_evidence_when_no_sentence_overlaps_query_at_all():
    # Mirrors the exact bug pattern from results/answer_quality_en_native.md:
    # a topically-similar (high embedding score), on-topic passage that
    # contains no sentence answering the query used to be returned WHOLE as
    # if it were the answer. It must now short-circuit before generation
    # ever produces a candidate "answer" to grade.
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored(
                    "Manhattan Project scientists gathered at Los Alamos. "
                    "They worked for several years on nuclear physics research.",
                    0.85,
                )
            ]
        )
    )
    result = _run("immediate impact of successful test", retriever)

    assert result.outcome == RagOutcome.INSUFFICIENT_EVIDENCE
    assert result.answer is None
    assert result.refusal_reason is not None
    assert "no sentence" in result.refusal_reason
    assert [t.stage for t in result.timings] == [
        "guardrail_unsafe",
        "embed",
        "retrieve",
        "guardrail_offtopic",
        "generate",
    ]


def test_total_ms_is_nonnegative_sum():
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9)]))
    result = _run("what is a corporation", retriever)
    assert result.total_ms >= 0.0
