"""Hot-path RAG orchestrator (spec §5, scoped to this phase): explicit typed
stages, per-stage timing, explicit typed refusal exits. No model calls in the
timed path — embedding is a small local model, everything else is heuristic or
extractive. Collaborators (embedder, retriever) are injected so the branching
logic is unit-testable with fakes."""

import time
from contextlib import contextmanager

from speculative_rag.generation import select_answer_span
from speculative_rag.guardrails import check_unsafe_input, groundedness_score, is_off_topic
from speculative_rag.pipeline_schemas import Answer, PipelineResult, RagOutcome, StageTiming


class _Timer:
    def __init__(self) -> None:
        self.timings: list[StageTiming] = []

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.timings.append(StageTiming(stage=name, elapsed_ms=elapsed_ms))


def run_pipeline(
    query_text: str,
    embedder,
    retriever,
    *,
    off_topic_threshold: float,
    groundedness_threshold: float,
    top_k: int,
) -> PipelineResult:
    timer = _Timer()

    with timer.stage("guardrail_unsafe"):
        unsafe_reason = check_unsafe_input(query_text)
    if unsafe_reason is not None:
        return PipelineResult(
            outcome=RagOutcome.UNSAFE_INPUT_REFUSED,
            refusal_reason=unsafe_reason,
            timings=timer.timings,
        )

    with timer.stage("embed"):
        query_vector = embedder.embed_query(query_text)

    with timer.stage("retrieve"):
        retrieval = retriever.search(query_vector, k=top_k)

    top = retrieval.top
    top_score = retrieval.top_score

    with timer.stage("guardrail_offtopic"):
        off_topic = top is None or is_off_topic(top_score, off_topic_threshold)
    if off_topic:
        return PipelineResult(
            outcome=RagOutcome.OFF_TOPIC_REFUSED,
            refusal_reason=(
                f"top retrieval score {top_score:.3f} below off-topic threshold {off_topic_threshold:.3f}"
            ),
            top_score=top_score,
            timings=timer.timings,
        )

    with timer.stage("generate"):
        span = select_answer_span(query_text, top.chunk.text)

    with timer.stage("guardrail_groundedness"):
        ground = groundedness_score(span, top.chunk.text)
    if ground < groundedness_threshold:
        return PipelineResult(
            outcome=RagOutcome.UNGROUNDED_REFUSED,
            refusal_reason=(
                f"groundedness {ground:.3f} below threshold {groundedness_threshold:.3f}"
            ),
            top_score=top_score,
            timings=timer.timings,
        )

    answer = Answer(
        text=span,
        source_passage_id=top.chunk.source_passage_id,
        source_chunk_id=top.chunk.chunk_id,
        retrieval_score=top_score,
        groundedness_score=ground,
    )
    return PipelineResult(
        outcome=RagOutcome.ANSWERED,
        answer=answer,
        top_score=top_score,
        timings=timer.timings,
    )
