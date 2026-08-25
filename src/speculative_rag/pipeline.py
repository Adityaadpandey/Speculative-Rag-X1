"""The hot-path RAG orchestrator: typed stages, per-stage timing, and typed
refusal exits. Nothing here calls a model directly — "embedding" is just
BM25 sparse term counting, not a neural model in-process, which is what
lets the deployed webapp run on a free 512MB host. Everything past that is
heuristic or extractive. The embedder and retriever are injected so we can
unit test the branching with fakes instead of a real index."""

import time
from contextlib import contextmanager

from speculative_rag.answer_type import expected_answer_type, span_satisfies_expected_type
from speculative_rag.generation import complete_answer_from_evidence, select_best_evidence
from speculative_rag.guardrails import (
    check_unsafe_input,
    groundedness_score,
    is_off_topic,
    plausible_extractive_answer,
)
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
    evidence_top_n: int = 10,
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
    num_retrieved = len(retrieval.scored_chunks)

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
            num_retrieved=num_retrieved,
            retrieved_chunks=retrieval.scored_chunks,
        )

    evidence_candidates = retrieval.scored_chunks[:evidence_top_n]
    with timer.stage("generate"):
        best = select_best_evidence(query_text, evidence_candidates)
    if best is None:
        return PipelineResult(
            outcome=RagOutcome.INSUFFICIENT_EVIDENCE,
            refusal_reason=(
                f"no sentence among the top {len(evidence_candidates)} retrieved passages "
                "shares vocabulary with the query text"
            ),
            top_score=top_score,
            timings=timer.timings,
            num_retrieved=num_retrieved,
            num_evidence_candidates=len(evidence_candidates),
            retrieved_chunks=retrieval.scored_chunks,
        )
    chosen, span, raw_sentence, matched_tokens = best
    # If extraction selected a terse prefix such as "India is a peninsula",
    # expand it to the complete source sentence before showing it to the user.
    # This remains grounded because every added token comes verbatim from the
    # selected passage.
    span = complete_answer_from_evidence(span, [chosen.chunk.text])

    with timer.stage("guardrail_groundedness"):
        ground = groundedness_score(span, chosen.chunk.text)
    if ground < groundedness_threshold:
        return PipelineResult(
            outcome=RagOutcome.UNGROUNDED_REFUSED,
            refusal_reason=(
                f"groundedness {ground:.3f} below threshold {groundedness_threshold:.3f}"
            ),
            top_score=top_score,
            timings=timer.timings,
            num_retrieved=num_retrieved,
            num_evidence_candidates=len(evidence_candidates),
            retrieved_chunks=retrieval.scored_chunks,
        )

    with timer.stage("guardrail_answer_type"):
        expected_type = expected_answer_type(query_text)
        type_satisfied = span_satisfies_expected_type(expected_type, span)
    if not type_satisfied:
        return PipelineResult(
            outcome=RagOutcome.INSUFFICIENT_EVIDENCE,
            refusal_reason=(
                f"extracted span does not contain the {expected_type} answer expected by the query"
            ),
            top_score=top_score,
            timings=timer.timings,
            num_retrieved=num_retrieved,
            num_evidence_candidates=len(evidence_candidates),
            retrieved_chunks=retrieval.scored_chunks,
        )

    if not plausible_extractive_answer(query_text, span, matched_tokens, chosen.chunk.language):
        return PipelineResult(
            outcome=RagOutcome.INSUFFICIENT_EVIDENCE,
            refusal_reason="retrieved text is topical but does not contain a complete answer",
            top_score=top_score,
            timings=timer.timings,
            num_retrieved=num_retrieved,
            num_evidence_candidates=len(evidence_candidates),
            retrieved_chunks=retrieval.scored_chunks,
        )

    answer = Answer(
        text=span,
        source_passage_id=chosen.chunk.source_passage_id,
        source_chunk_id=chosen.chunk.chunk_id,
        retrieval_score=chosen.score,
        groundedness_score=ground,
        raw_sentence=raw_sentence,
        matched_tokens=matched_tokens,
    )
    return PipelineResult(
        outcome=RagOutcome.ANSWERED,
        answer=answer,
        top_score=top_score,
        timings=timer.timings,
        num_retrieved=num_retrieved,
        num_evidence_candidates=len(evidence_candidates),
        retrieved_chunks=retrieval.scored_chunks,
    )
