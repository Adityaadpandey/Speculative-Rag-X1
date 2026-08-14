from speculative_rag.pipeline_schemas import (
    Answer,
    PipelineResult,
    RagOutcome,
    RetrievalResult,
    ScoredChunk,
    StageTiming,
)
from speculative_rag.schemas import Chunk


def _chunk(pid="p1"):
    return Chunk(
        chunk_id=f"{pid}:native:0",
        text="a corporation is a company",
        language="en",
        query_type="DESCRIPTION",
        source_passage_id=pid,
        strategy="native",
    )


def test_retrieval_result_top_and_top_score():
    result = RetrievalResult(
        scored_chunks=[
            ScoredChunk(chunk=_chunk("p1"), score=0.9),
            ScoredChunk(chunk=_chunk("p2"), score=0.4),
        ]
    )
    assert result.top.chunk.source_passage_id == "p1"
    assert result.top_score == 0.9


def test_retrieval_result_empty_top_is_none_and_score_zero():
    result = RetrievalResult(scored_chunks=[])
    assert result.top is None
    assert result.top_score == 0.0


def test_pipeline_result_total_ms_sums_timings():
    result = PipelineResult(
        outcome=RagOutcome.ANSWERED,
        answer=Answer(
            text="a corporation",
            source_passage_id="p1",
            source_chunk_id="p1:native:0",
            retrieval_score=0.9,
            groundedness_score=1.0,
        ),
        top_score=0.9,
        timings=[StageTiming(stage="embed", elapsed_ms=3.0), StageTiming(stage="retrieve", elapsed_ms=7.5)],
    )
    assert result.total_ms == 10.5
    assert result.outcome.value == "answered"


def test_pipeline_result_defaults_are_safe():
    result = PipelineResult(outcome=RagOutcome.UNSAFE_INPUT_REFUSED, refusal_reason="nope")
    assert result.answer is None
    assert result.timings == []
    assert result.total_ms == 0.0
