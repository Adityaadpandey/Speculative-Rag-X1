"""Recovery control flow: what actually happens after the fast hot-path
answer (pipeline.run_pipeline) fails validation.

Before this module existed, the fast extractive answer, the LLM groundedness
check, the LLM rerank, and the LLM multi-passage synthesis were four
separate operations (see webapp/main.py's old /api/reflect, /api/rerank,
/api/synthesize routes) stitched together only by the browser firing one
fetch after another and painting each result into its own side panel. The
reranked passage never actually became the evidence for a new generation
call -- the UI just showed it next to the original (possibly ungrounded)
answer, which stayed on screen marked "ANSWERED" regardless.

resolve_recovery (with resolve_answer as a synchronous wrapper that also
runs the fast path itself) is now the single owner of the real control flow:

    generate_initial_answer (fast, extractive, ~ms)
        -> validate_initial_answer (LLM groundedness/answerability)
            -> valid: return ANSWERED
            -> invalid: rerank_passages (LLM, structured selection)
                -> nothing relevant found: return INSUFFICIENT_EVIDENCE
                -> generate_answer(query, selected evidence) (LLM)
                    -> validate_regenerated_answer (LLM, same check as above)
                        -> valid: return ANSWERED
                        -> invalid: multi_passage_synthesis (LLM)
                            -> validate() -> return ANSWERED or FALLBACK

Only the fast path (embed/retrieve/extract) has to hit the ~200ms latency
budget; every LLM call here is deliberately outside that budget and only
runs when the cheaper stage before it actually failed -- rerank, regenerate,
and synthesis never run for a query the fast path already got right.
"""

import time
from typing import Literal

from pydantic import BaseModel

from speculative_rag.answer_status import FinalAnswerStatus, combine_verdict
from speculative_rag.generation import prefilter_rerank_candidates
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import PipelineResult, RagOutcome
from speculative_rag.reflect import (
    RerankCandidate,
    check_groundedness_llm,
    rerank_and_generate_llm,
    synthesize_answer_llm,
)


class AnswerOutcome(BaseModel):
    final_status: FinalAnswerStatus
    answer_text: str | None
    source_passage_ids: list[str]
    reason: str
    # Which stage actually produced answer_text, set explicitly at each of
    # the three ANSWERED return points below and None everywhere else -- so
    # the UI never has to infer this from which timeline steps ran (fragile:
    # rerank can run but the answer still come from synthesis, or rerank can
    # run and fail while the original extractive answer was already valid).
    answer_engine: Literal["extractive", "claude_regenerated", "claude_synthesized"] | None = None

    # Instrumentation -- trace the actual data flow instead of just logging what the UI shows.
    initial_answer_valid: bool
    initial_answerable: bool
    rerank_triggered: bool
    rerank_latency_ms: float | None
    selected_source_ids: list[str]
    reranked_answer_generated: bool
    regenerated_answer_valid: bool | None
    regenerated_answerable: bool | None
    synthesis_triggered: bool
    total_end_to_end_latency_ms: float

    pipeline_result: PipelineResult


def resolve_answer(
    query_text: str,
    embedder,
    retriever,
    *,
    off_topic_threshold: float,
    groundedness_threshold: float,
    top_k: int,
    evidence_top_n: int = 5,
    check_groundedness=None,
    rerank_and_generate=None,
    synthesize_answer=None,
    on_stage=None,
    prefilter_enabled: bool = False,
    prefilter_lexical_top_n: int = 8,
    prefilter_retrieval_reserve: int = 5,
) -> AnswerOutcome:
    """Synchronous convenience wrapper: runs the fast path (run_pipeline)
    and the recovery chain (resolve_recovery) back to back. Used by CLI/eval
    scripts and by tests that don't care about the two phases being on
    different sides of an HTTP response. The webapp doesn't use this one --
    see webapp/main.py's post_answer, which returns the fast path
    immediately and kicks resolve_recovery off afterward as a background
    task, so the ~200ms latency budget never has to wait on the LLM calls
    below."""
    pipeline_result = run_pipeline(
        query_text,
        embedder,
        retriever,
        off_topic_threshold=off_topic_threshold,
        groundedness_threshold=groundedness_threshold,
        top_k=top_k,
        evidence_top_n=evidence_top_n,
    )
    return resolve_recovery(
        query_text,
        pipeline_result,
        check_groundedness=check_groundedness,
        rerank_and_generate=rerank_and_generate,
        synthesize_answer=synthesize_answer,
        on_stage=on_stage,
        prefilter_enabled=prefilter_enabled,
        prefilter_lexical_top_n=prefilter_lexical_top_n,
        prefilter_retrieval_reserve=prefilter_retrieval_reserve,
    )


def resolve_recovery(
    query_text: str,
    pipeline_result: PipelineResult,
    *,
    check_groundedness=None,
    rerank_and_generate=None,
    synthesize_answer=None,
    prefilter_enabled: bool = False,
    prefilter_lexical_top_n: int = 8,
    prefilter_retrieval_reserve: int = 5,
    on_stage=None,
) -> AnswerOutcome:
    """The LLM validation/rerank+regenerate/synthesis chain, taking an
    already-computed PipelineResult (the fast path's output) rather than
    running it itself. Split out from resolve_answer so the webapp can run
    the fast path synchronously within the latency budget and hand this off
    to a background task afterward -- every call in here is a real network
    round trip to an LLM, deliberately kept off that budget.

    rerank_and_generate does passage selection AND answer generation in one
    call (reflect.rerank_and_generate_llm) rather than two separate round
    trips -- check_groundedness stays a fully separate, independent
    verification call both before and after it.

    on_stage(stage: str), if given, gets called right before each LLM round
    trip starts ("validating", "reranking", "synthesizing") so a caller
    running this in the background (webapp/main.py) can surface live
    progress instead of one opaque wait -- it's purely a progress signal,
    never consulted for control flow."""
    # Looked up as module globals rather than bound as literal defaults so
    # callers (webapp/main.py) can monkeypatch
    # speculative_rag.answer_orchestrator.check_groundedness_llm etc. in
    # tests, the same pattern webapp/main.py already uses for reflect.py.
    check_groundedness = check_groundedness or check_groundedness_llm
    rerank_and_generate = rerank_and_generate or rerank_and_generate_llm
    synthesize_answer = synthesize_answer or synthesize_answer_llm
    on_stage = on_stage or (lambda stage: None)

    start = time.perf_counter()

    def elapsed_ms() -> float:
        return (time.perf_counter() - start) * 1000.0

    def terminal(
        final_status: FinalAnswerStatus,
        reason: str,
        *,
        initial_answer_valid: bool = False,
        initial_answerable: bool = False,
        rerank_triggered: bool = False,
        rerank_latency_ms: float | None = None,
        selected_source_ids: list[str] | None = None,
        reranked_answer_generated: bool = False,
        regenerated_answer_valid: bool | None = None,
        regenerated_answerable: bool | None = None,
        synthesis_triggered: bool = False,
        answer_text: str | None = None,
        source_passage_ids: list[str] | None = None,
        answer_engine: Literal["extractive", "claude_regenerated", "claude_synthesized"] | None = None,
    ) -> AnswerOutcome:
        return AnswerOutcome(
            final_status=final_status,
            answer_text=answer_text,
            source_passage_ids=source_passage_ids or [],
            reason=reason,
            answer_engine=answer_engine,
            initial_answer_valid=initial_answer_valid,
            initial_answerable=initial_answerable,
            rerank_triggered=rerank_triggered,
            rerank_latency_ms=rerank_latency_ms,
            selected_source_ids=selected_source_ids or [],
            reranked_answer_generated=reranked_answer_generated,
            regenerated_answer_valid=regenerated_answer_valid,
            regenerated_answerable=regenerated_answerable,
            synthesis_triggered=synthesis_triggered,
            total_end_to_end_latency_ms=elapsed_ms(),
            pipeline_result=pipeline_result,
        )

    if pipeline_result.outcome == RagOutcome.UNSAFE_INPUT_REFUSED:
        return terminal(FinalAnswerStatus.NOT_ANSWERED, pipeline_result.refusal_reason or "unsafe input")

    if pipeline_result.outcome == RagOutcome.OFF_TOPIC_REFUSED:
        return terminal(FinalAnswerStatus.INSUFFICIENT_EVIDENCE, pipeline_result.refusal_reason or "off-topic")

    # Reuse the candidate pool run_pipeline already retrieved
    # (PipelineResult.retrieved_chunks) instead of paying for a second
    # embed+retrieve round trip against the same query -- retrieve is the
    # dominant cost on this corpus (local-mode Qdrant, seconds per call), so
    # doing it twice was roughly doubling total latency for nothing. We need
    # this both to find the real source text for the groundedness check
    # below (the extractive answer could've come from any of the
    # evidence_top_n candidates, not necessarily rank-1) and to feed
    # rerank/synthesis.
    candidates = pipeline_result.retrieved_chunks

    initial_answer_valid = False
    initial_answerable = False

    if pipeline_result.outcome == RagOutcome.ANSWERED:
        answer = pipeline_result.answer
        assert answer is not None
        source_chunk = next((c for c in candidates if c.chunk.chunk_id == answer.source_chunk_id), None)
        source_text = source_chunk.chunk.text if source_chunk is not None else answer.text

        on_stage("validating")
        verdict = check_groundedness(query_text, answer.text, source_text)
        initial_answerable = verdict.answers_query
        status, reason = combine_verdict(RagOutcome.ANSWERED, verdict)
        if status == FinalAnswerStatus.ANSWERED:
            return terminal(
                FinalAnswerStatus.ANSWERED,
                reason,
                initial_answer_valid=True,
                initial_answerable=True,
                answer_text=answer.text,
                source_passage_ids=[answer.source_passage_id],
                answer_engine="extractive",
            )
        # Initial answer didn't pass validation -- fall through to recovery below.

    if not candidates:
        return terminal(
            FinalAnswerStatus.INSUFFICIENT_EVIDENCE,
            "no candidates retrieved",
            initial_answer_valid=initial_answer_valid,
            initial_answerable=initial_answerable,
        )

    # Optional non-LLM prefilter narrowing the pool before it's handed to the
    # LLM rerank+regenerate call below -- ships disabled by default (see
    # config.SETTINGS.rerank_prefilter_enabled); prefilter_rerank_candidates
    # itself always falls back to the full pool when it can't find any
    # meaningful lexical signal to filter by, so enabling this can narrow the
    # LLM's input but never expand it beyond what was already retrieved.
    prefilter_candidates = candidates
    if prefilter_enabled:
        prefilter_candidates = prefilter_rerank_candidates(
            query_text,
            candidates,
            lexical_top_n=prefilter_lexical_top_n,
            retrieval_reserve=prefilter_retrieval_reserve,
        )

    # Rerank + regenerate in one call: look across the (possibly prefiltered)
    # retrieved pool, not just embedding rank-1, find which passage(s)
    # actually contain the answer, and answer directly from them -- this is
    # the step that was missing before this module existed (see the module
    # docstring).
    rerank_candidates = [RerankCandidate(source_id=c.chunk.chunk_id, text=c.chunk.text) for c in prefilter_candidates]
    on_stage("reranking")
    rerank_start = time.perf_counter()
    rerank_verdict = rerank_and_generate(query_text, rerank_candidates)
    rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000.0
    selected_source_ids = [s.source_id for s in rerank_verdict.selected_sources]

    if not (rerank_verdict.selection_found and selected_source_ids):
        return terminal(
            FinalAnswerStatus.INSUFFICIENT_EVIDENCE,
            rerank_verdict.reasoning or "none of the retrieved passages answer the question",
            initial_answer_valid=initial_answer_valid,
            initial_answerable=initial_answerable,
            rerank_triggered=True,
            rerank_latency_ms=rerank_latency_ms,
        )

    by_chunk_id = {c.chunk.chunk_id: c for c in candidates}
    evidence_chunks = [by_chunk_id[sid] for sid in selected_source_ids if sid in by_chunk_id]
    evidence_texts = [c.chunk.text for c in evidence_chunks]

    reranked_answer_generated = True
    regenerated_answer_valid = False
    regenerated_answerable = False

    if rerank_verdict.answerable and rerank_verdict.answer_text:
        combined_source_text = " ".join(evidence_texts)
        on_stage("verifying")
        verdict2 = check_groundedness(query_text, rerank_verdict.answer_text, combined_source_text)
        regenerated_answerable = verdict2.answers_query
        status2, reason2 = combine_verdict(RagOutcome.ANSWERED, verdict2)
        regenerated_answer_valid = status2 == FinalAnswerStatus.ANSWERED
        if regenerated_answer_valid:
            return terminal(
                FinalAnswerStatus.ANSWERED,
                reason2,
                initial_answer_valid=initial_answer_valid,
                initial_answerable=initial_answerable,
                rerank_triggered=True,
                rerank_latency_ms=rerank_latency_ms,
                selected_source_ids=selected_source_ids,
                reranked_answer_generated=True,
                regenerated_answer_valid=True,
                regenerated_answerable=regenerated_answerable,
                answer_text=rerank_verdict.answer_text,
                source_passage_ids=[c.chunk.source_passage_id for c in evidence_chunks],
                answer_engine="claude_regenerated",
            )

    # Regeneration from the reranked evidence still didn't pass validation
    # (or the generator itself decided the evidence wasn't enough) -- fall
    # back further to multi-passage synthesis across the full candidate pool.
    on_stage("synthesizing")
    synthesis_verdict = synthesize_answer(query_text, [c.chunk.text for c in candidates])
    if synthesis_verdict.answerable and synthesis_verdict.answer_text:
        synth_sources = [candidates[i] for i in synthesis_verdict.source_indices if 0 <= i < len(candidates)]
        combined_source_text = " ".join(c.chunk.text for c in synth_sources) or " ".join(
            c.chunk.text for c in candidates
        )
        on_stage("verifying")
        verdict3 = check_groundedness(query_text, synthesis_verdict.answer_text, combined_source_text)
        status3, reason3 = combine_verdict(RagOutcome.ANSWERED, verdict3)
        if status3 == FinalAnswerStatus.ANSWERED:
            return terminal(
                FinalAnswerStatus.ANSWERED,
                reason3,
                initial_answer_valid=initial_answer_valid,
                initial_answerable=initial_answerable,
                rerank_triggered=True,
                rerank_latency_ms=rerank_latency_ms,
                selected_source_ids=selected_source_ids,
                reranked_answer_generated=reranked_answer_generated,
                regenerated_answer_valid=regenerated_answer_valid,
                regenerated_answerable=regenerated_answerable,
                synthesis_triggered=True,
                answer_text=synthesis_verdict.answer_text,
                source_passage_ids=[c.chunk.source_passage_id for c in synth_sources],
                answer_engine="claude_synthesized",
            )
        reason = f"multi-passage synthesis produced an answer but it failed validation: {reason3}"
    else:
        reason = synthesis_verdict.reasoning or "no passage, alone or combined, answers the question"

    return terminal(
        FinalAnswerStatus.FALLBACK,
        reason,
        initial_answer_valid=initial_answer_valid,
        initial_answerable=initial_answerable,
        rerank_triggered=True,
        rerank_latency_ms=rerank_latency_ms,
        selected_source_ids=selected_source_ids,
        reranked_answer_generated=reranked_answer_generated,
        regenerated_answer_valid=regenerated_answer_valid,
        regenerated_answerable=regenerated_answerable,
        synthesis_triggered=True,
    )
