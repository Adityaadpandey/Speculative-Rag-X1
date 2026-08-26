"""Fast adapter for rag-local-eval-loop's generator interface.

The evaluator measures ``generate_answer`` synchronously. The production
webapp deliberately does not: it returns the extractive answer from
``run_pipeline`` and schedules LLM recovery in a background task. Calling
that chain here would turn the sub-200ms hot path into a 10--20 second
network request and contradict the product SLA.

The harness supplies already-retrieved results from its isolated FAISS index,
so this module reproduces the production post-retrieval hot path. Set
``EVAL_GENERATION_MODE=recovery`` for a deliberate synchronous comparison
that continues through ``resolve_recovery`` and the real Anthropic Claude
Haiku calls. ``EVAL_RECOVERY_PROVIDER=openai`` keeps the same chain but uses
the project's OpenAI callback adapter when Anthropic credits are unavailable.
"""
import os
import time
from dataclasses import dataclass

from speculative_rag.answer_orchestrator import resolve_recovery
from speculative_rag.answer_status import FinalAnswerStatus
from speculative_rag.answer_type import expected_answer_type, span_satisfies_expected_type
from speculative_rag.config import SETTINGS
from speculative_rag.generation import complete_answer_from_evidence, select_best_evidence
from speculative_rag.guardrails import groundedness_score, plausible_extractive_answer
from speculative_rag.pipeline_schemas import Answer, PipelineResult, RagOutcome, ScoredChunk
from speculative_rag.schemas import Chunk


_GENERATION_MODE = os.environ.get("EVAL_GENERATION_MODE", "local").lower()
_RECOVERY_PROVIDER = os.environ.get("EVAL_RECOVERY_PROVIDER", "anthropic").lower()
GENERATION_BACKEND = "recovery-chain" if _GENERATION_MODE == "recovery" else "local"
GENERATION_MODEL = (
    "claude-haiku-4-5-20251001"
    if _GENERATION_MODE == "recovery" and _RECOVERY_PROVIDER == "anthropic"
    else "openai-recovery-chain"
    if _GENERATION_MODE == "recovery"
    else "extractive"
)


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


def _to_scored_chunks(results: list) -> list[ScoredChunk]:
    scored_chunks = []
    for rank, result in enumerate(results):
        chunk = Chunk(
            chunk_id=str(result.source),
            text=result.text,
            language="en",
            query_type="eval",
            source_passage_id=str(result.source),
            strategy="eval",
        )
        # Results arrive ranked best-first. Preserve their actual cosine score
        # for select_best_evidence's retrieval-score tie-break.
        score = float(getattr(result, "score", len(results) - rank))
        scored_chunks.append(ScoredChunk(chunk=chunk, score=score))
    return scored_chunks


def _fast_path(query_text: str, candidates: list[ScoredChunk]) -> PipelineResult:
    """Mirror run_pipeline after retrieval, without any network call."""
    if not candidates:
        return PipelineResult(outcome=RagOutcome.INSUFFICIENT_EVIDENCE, refusal_reason="no candidates retrieved")

    best = select_best_evidence(query_text, candidates)
    if best is None:
        return PipelineResult(
            outcome=RagOutcome.INSUFFICIENT_EVIDENCE,
            refusal_reason="no retrieved sentence shares meaningful vocabulary with the query",
            retrieved_chunks=candidates,
        )

    chosen, span, raw_sentence, matched_tokens = best
    span = complete_answer_from_evidence(span, [chosen.chunk.text])
    ground = groundedness_score(span, chosen.chunk.text)
    if not plausible_extractive_answer(query_text, span, matched_tokens, chosen.chunk.language):
        return PipelineResult(
            outcome=RagOutcome.INSUFFICIENT_EVIDENCE,
            refusal_reason="retrieved text is topical but does not contain a complete answer",
            retrieved_chunks=candidates,
        )
    if ground < SETTINGS.groundedness_threshold:
        return PipelineResult(
            outcome=RagOutcome.UNGROUNDED_REFUSED,
            refusal_reason=f"groundedness {ground:.3f} below threshold {SETTINGS.groundedness_threshold:.3f}",
            retrieved_chunks=candidates,
        )

    expected_type = expected_answer_type(query_text)
    if not span_satisfies_expected_type(expected_type, span):
        return PipelineResult(
            outcome=RagOutcome.INSUFFICIENT_EVIDENCE,
            refusal_reason=f"extracted span does not contain the {expected_type} answer expected by the query",
            retrieved_chunks=candidates,
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
    return PipelineResult(outcome=RagOutcome.ANSWERED, answer=answer, retrieved_chunks=candidates)


def generate_answer(query: str, results: list) -> GeneratedAnswer:
    """Run either the measured hot path or the explicit recovery comparison.

    ``recovery`` is intentionally synchronous here because the evaluator must
    measure the real LLM chain. The production webapp keeps the same chain in
    a background task and therefore does not put this latency on its SLA.
    """
    t0 = time.perf_counter()
    pipeline_result = _fast_path(query, _to_scored_chunks(results))

    if _GENERATION_MODE == "recovery":
        recovery_kwargs = {
            "prefilter_enabled": SETTINGS.rerank_prefilter_enabled,
            "prefilter_lexical_top_n": SETTINGS.rerank_prefilter_lexical_top_n,
            "prefilter_retrieval_reserve": SETTINGS.rerank_prefilter_retrieval_reserve,
            "dense_rerank_enabled": SETTINGS.dense_rerank_enabled,
        }
        if _RECOVERY_PROVIDER == "openai":
            # The control flow is still the same resolve_recovery chain; this
            # provider switch is useful when the configured Anthropic account
            # has no remaining credits. The default remains reflect.py/Claude.
            from speculative_rag.openai_reflect import (
                check_groundedness,
                rerank_and_generate,
                synthesize_answer,
            )

            recovery_kwargs.update(
                check_groundedness=check_groundedness,
                rerank_and_generate=rerank_and_generate,
                synthesize_answer=synthesize_answer,
            )
        outcome = resolve_recovery(query, pipeline_result, **recovery_kwargs)
        generation_ms = (time.perf_counter() - t0) * 1000.0
        answered = outcome.final_status == FinalAnswerStatus.ANSWERED and bool(outcome.answer_text)
        return GeneratedAnswer(
            text=outcome.answer_text if answered else "",
            grounded=answered,
            generation_ms=generation_ms,
            model=outcome.answer_engine or "none",
        )

    generation_ms = (time.perf_counter() - t0) * 1000.0
    answer = pipeline_result.answer
    answered = pipeline_result.outcome == RagOutcome.ANSWERED and answer is not None
    return GeneratedAnswer(
        text=answer.text if answered else "",
        grounded=answered,
        generation_ms=generation_ms,
        model="extractive" if answered else "none",
    )
