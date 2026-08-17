from speculative_rag.answer_status import FinalAnswerStatus, combine_verdict
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.reflect import ReflectVerdict


def _verdict(**overrides):
    base = dict(grounded=True, answers_query=True, complete=True, relevant=True, reasoning="ok")
    base.update(overrides)
    return ReflectVerdict(**base)


def test_pending_when_answered_outcome_has_no_verdict_yet():
    status, reason = combine_verdict(RagOutcome.ANSWERED, None)
    assert status is None
    assert "awaiting" in reason


def test_all_pass_resolves_to_answered():
    status, _ = combine_verdict(RagOutcome.ANSWERED, _verdict())
    assert status == FinalAnswerStatus.ANSWERED


def test_case3_verbatim_grounded_but_does_not_answer_query_is_not_answered():
    # Adversarial case: the answer copies the passage exactly (grounded=1.0
    # in the old lexical sense) but never states what the query asked for.
    # This must not resolve to ANSWERED.
    verdict = _verdict(answers_query=False)
    status, reason = combine_verdict(RagOutcome.ANSWERED, verdict)
    assert status == FinalAnswerStatus.NOT_ANSWERED
    assert "does not answer" in reason


def test_case4_answers_query_but_has_unsupported_claim_is_ungrounded():
    # Adversarial case: answers the question correctly but adds one
    # unsupported fact. Faithfulness fails even though answerability passes.
    verdict = _verdict(grounded=False)
    status, _ = combine_verdict(RagOutcome.ANSWERED, verdict)
    assert status == FinalAnswerStatus.UNGROUNDED


def test_incomplete_answer_is_not_answered_even_if_grounded_and_on_topic():
    verdict = _verdict(complete=False)
    status, reason = combine_verdict(RagOutcome.ANSWERED, verdict)
    assert status == FinalAnswerStatus.NOT_ANSWERED
    assert "missing" in reason


def test_irrelevant_context_dump_is_not_answered():
    verdict = _verdict(relevant=False)
    status, _ = combine_verdict(RagOutcome.ANSWERED, verdict)
    assert status == FinalAnswerStatus.NOT_ANSWERED


def test_off_topic_outcome_maps_to_insufficient_evidence():
    status, _ = combine_verdict(RagOutcome.OFF_TOPIC_REFUSED, None)
    assert status == FinalAnswerStatus.INSUFFICIENT_EVIDENCE


def test_insufficient_evidence_outcome_passes_through():
    # Adversarial case: retrieved passage is on-topic but doesn't contain
    # the answer. That's handled entirely in the hot path already; this
    # just confirms combine_verdict passes that outcome straight through.
    status, _ = combine_verdict(RagOutcome.INSUFFICIENT_EVIDENCE, None)
    assert status == FinalAnswerStatus.INSUFFICIENT_EVIDENCE


def test_ungrounded_refused_outcome_maps_to_ungrounded():
    status, _ = combine_verdict(RagOutcome.UNGROUNDED_REFUSED, None)
    assert status == FinalAnswerStatus.UNGROUNDED


def test_unsafe_input_refused_maps_to_not_answered():
    status, _ = combine_verdict(RagOutcome.UNSAFE_INPUT_REFUSED, None)
    assert status == FinalAnswerStatus.NOT_ANSWERED
