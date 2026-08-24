"""Adapter satisfying rag-local-eval-loop's generator interface
(generate_answer(query, results) -> answer with .text/.grounded/
.generation_ms/.model) by reusing this project's REAL production answer
pipeline end to end: the extractive fast path (speculative_rag.pipeline's
post-retrieval logic) plus the full LLM recovery chain
(speculative_rag.answer_orchestrator.resolve_recovery -- groundedness
validation, rerank+regenerate, multi-passage synthesis) with the same
prompts and control flow as webapp/main.py's /api/answer route, but backed
by OpenAI instead of Anthropic (see speculative_rag/openai_reflect.py,
the same module the deployed webapp itself now runs on) so eval runs
don't draw down the project's Anthropic credits.

The one piece NOT reproduced here is embed+retrieve: run_pipeline normally
does that against the real BM25/Qdrant index, but this eval harness supplies
its own already-retrieved `results` (from its own throwaway dense FAISS
index over app.embedder -- see TARGET_INTERFACE.md). So this module
replicates only run_pipeline's POST-retrieval steps (extraction,
groundedness threshold, answer-type check) to build the same PipelineResult
shape resolve_recovery expects, then hands off to the real recovery chain
unchanged. Off-topic guardrail is intentionally not replicated: it's gated
on a raw BM25 top score (speculative_rag.config.SETTINGS.off_topic_threshold),
which has no equivalent for the harness's own dense-similarity results.

This makes real OpenAI API calls (OPENAI_API_KEY) whenever the fast
extractive path doesn't validate on its own -- so generation_ms and .model
here reflect real recovery-chain cost, not just the sub-millisecond
extractive stage.
"""
import time
from dataclasses import dataclass

from speculative_rag.answer_orchestrator import resolve_recovery
from speculative_rag.answer_status import FinalAnswerStatus
from speculative_rag.answer_type import expected_answer_type, span_satisfies_expected_type
from speculative_rag.config import SETTINGS
from speculative_rag.generation import complete_answer_from_evidence, select_best_evidence
from speculative_rag.guardrails import groundedness_score
from speculative_rag.pipeline_schemas import Answer, PipelineResult, RagOutcome, ScoredChunk
from speculative_rag.schemas import Chunk

from speculative_rag.openai_reflect import check_groundedness, rerank_and_generate, synthesize_answer

GENERATION_BACKEND = "recovery-chain"
GENERATION_MODEL = "speculative-rag-answer-orchestrator"


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
        # Results already arrive ranked best-first by the eval loop's own
        # FAISS search, so a simple descending rank score reproduces that
        # order for select_best_evidence's retrieval-score tiebreak.
        scored_chunks.append(ScoredChunk(chunk=chunk, score=len(results) - rank))
    return scored_chunks


def _fast_path(query_text: str, candidates: list[ScoredChunk]) -> PipelineResult:
    """Mirrors speculative_rag.pipeline.run_pipeline's logic from
    "evidence_candidates = ..." onward -- everything after retrieval, which
    the eval harness already did for us. Skips the off-topic guardrail (see
    module docstring); groundedness and answer-type guardrails run exactly
    as in production."""
    if not candidates:
        return PipelineResult(outcome=RagOutcome.INSUFFICIENT_EVIDENCE, refusal_reason="no candidates retrieved")

    best = select_best_evidence(query_text, candidates)
    if best is None:
        return PipelineResult(
            outcome=RagOutcome.INSUFFICIENT_EVIDENCE,
            refusal_reason="no sentence among the retrieved passages shares vocabulary with the query text",
            retrieved_chunks=candidates,
        )
    chosen, span, raw_sentence, matched_tokens = best
    span = complete_answer_from_evidence(span, [chosen.chunk.text])

    ground = groundedness_score(span, chosen.chunk.text)
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
    t0 = time.perf_counter()
    candidates = _to_scored_chunks(results)
    pipeline_result = _fast_path(query, candidates)

    outcome = resolve_recovery(
        query,
        pipeline_result,
        check_groundedness=check_groundedness,
        rerank_and_generate=rerank_and_generate,
        synthesize_answer=synthesize_answer,
        prefilter_enabled=SETTINGS.rerank_prefilter_enabled,
        prefilter_lexical_top_n=SETTINGS.rerank_prefilter_lexical_top_n,
        prefilter_retrieval_reserve=SETTINGS.rerank_prefilter_retrieval_reserve,
        dense_rerank_enabled=SETTINGS.dense_rerank_enabled,
    )
    generation_ms = (time.perf_counter() - t0) * 1000.0

    return GeneratedAnswer(
        text=outcome.answer_text or "",
        grounded=outcome.final_status == FinalAnswerStatus.ANSWERED,
        generation_ms=generation_ms,
        model=outcome.answer_engine or "none",
    )
