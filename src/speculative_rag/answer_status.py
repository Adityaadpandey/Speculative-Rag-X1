"""Combines the fast, deterministic hot-path RagOutcome with the async,
four-dimension LLM ReflectVerdict into the single status actually shown to
the end user. Kept out of pipeline.py on purpose -- the ~4s LLM groundedness
call must never enter the synchronous critical path -- so this function is
pure and cheap, called by webapp/eval code once the async verdict resolves.

Core rule: RagOutcome.ANSWERED only means "the hot path produced a
candidate extractive answer that passed the cheap heuristic gates." It's
NOT a correctness verdict. A grounded=1.0, verbatim-from-source answer that
doesn't actually state what the query asked for has to resolve to
NOT_ANSWERED here, never ANSWERED."""

from enum import Enum

from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.reflect import ReflectVerdict


class FinalAnswerStatus(str, Enum):
    ANSWERED = "answered"
    NOT_ANSWERED = "not_answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNGROUNDED = "ungrounded"


_NON_ANSWERED_OUTCOME_STATUS: dict[RagOutcome, FinalAnswerStatus] = {
    RagOutcome.OFF_TOPIC_REFUSED: FinalAnswerStatus.INSUFFICIENT_EVIDENCE,
    RagOutcome.INSUFFICIENT_EVIDENCE: FinalAnswerStatus.INSUFFICIENT_EVIDENCE,
    RagOutcome.UNGROUNDED_REFUSED: FinalAnswerStatus.UNGROUNDED,
    RagOutcome.UNSAFE_INPUT_REFUSED: FinalAnswerStatus.NOT_ANSWERED,
}


def combine_verdict(
    outcome: RagOutcome, verdict: ReflectVerdict | None
) -> tuple[FinalAnswerStatus | None, str]:
    """Returns (status, reason). status is None only when outcome is
    ANSWERED and the async verdict hasn't resolved yet -- callers need to
    treat that as "pending," never display it as answered."""
    if outcome != RagOutcome.ANSWERED:
        status = _NON_ANSWERED_OUTCOME_STATUS[outcome]
        return status, f"hot path returned {outcome.value}, no candidate answer to verify"

    if verdict is None:
        return None, "awaiting async faithfulness/answerability verification"

    if not verdict.grounded:
        return FinalAnswerStatus.UNGROUNDED, f"unsupported claim(s) in the answer: {verdict.reasoning}"
    if not verdict.answers_query:
        return (
            FinalAnswerStatus.NOT_ANSWERED,
            f"grounded but does not answer the query: {verdict.reasoning}",
        )
    if not verdict.complete:
        return FinalAnswerStatus.NOT_ANSWERED, f"answer is missing required information: {verdict.reasoning}"
    if not verdict.relevant:
        return FinalAnswerStatus.NOT_ANSWERED, f"answer is not focused on the query: {verdict.reasoning}"

    return FinalAnswerStatus.ANSWERED, verdict.reasoning
