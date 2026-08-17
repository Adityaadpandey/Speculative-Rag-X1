import anthropic
import pytest

from speculative_rag.reflect import (
    ReflectVerdict,
    RerankVerdict,
    SynthesisVerdict,
    check_groundedness_llm,
    select_best_candidate_llm,
    synthesize_answer_llm,
)


def test_reflect_verdict_model_shape():
    verdict = ReflectVerdict(
        grounded=True,
        answers_query=True,
        complete=True,
        relevant=True,
        reasoning="The answer restates the source directly.",
    )
    assert verdict.grounded is True
    assert verdict.answers_query is True
    assert verdict.complete is True
    assert verdict.relevant is True
    assert "restates" in verdict.reasoning


def test_check_groundedness_llm_calls_sdk_with_expected_shape(monkeypatch):
    captured = {}

    class _FakeParsedMessage:
        parsed_output = ReflectVerdict(
            grounded=False,
            answers_query=False,
            complete=False,
            relevant=True,
            reasoning="Introduces an unsupported claim.",
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = check_groundedness_llm(
        "what country is Paris in",
        "Paris is in France.",
        "Paris, France's capital, sits on the Seine.",
    )

    assert verdict.grounded is False
    assert verdict.complete is False
    assert verdict.relevant is True
    assert captured["model"] == "claude-sonnet-5"
    assert captured["output_format"] is ReflectVerdict
    joined = " ".join(m["content"] for m in captured["messages"])
    assert "what country is Paris in" in joined
    assert "Paris is in France." in joined
    assert "Paris, France's capital, sits on the Seine." in joined


@pytest.mark.slow
def test_check_groundedness_llm_real_call():
    try:
        verdict = check_groundedness_llm(
            "what is the capital of France",
            "Paris is the capital of France.",
            "France's capital city is Paris, located on the Seine river.",
        )
    except anthropic.AnthropicError as exc:
        pytest.skip(f"Anthropic API not configured: {exc}")
    assert isinstance(verdict.grounded, bool)
    assert isinstance(verdict.answers_query, bool)
    assert isinstance(verdict.complete, bool)
    assert isinstance(verdict.relevant, bool)
    assert verdict.reasoning


def test_rerank_verdict_model_shape():
    verdict = RerankVerdict(best_index=1, reasoning="Candidate 1 directly names the capital.")
    assert verdict.best_index == 1
    assert "capital" in verdict.reasoning


def test_select_best_candidate_llm_calls_sdk_with_expected_shape(monkeypatch):
    captured = {}

    class _FakeParsedMessage:
        parsed_output = RerankVerdict(best_index=1, reasoning="Candidate 1 answers the question.")

    class _FakeMessages:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = select_best_candidate_llm(
        "what is the capital of France",
        ["Paris has many museums.", "The capital of France is Paris."],
    )

    assert verdict.best_index == 1
    assert captured["model"] == "claude-sonnet-5"
    assert captured["output_format"] is RerankVerdict
    joined = " ".join(m["content"] for m in captured["messages"])
    assert "what is the capital of France" in joined
    assert "[0] Paris has many museums." in joined
    assert "[1] The capital of France is Paris." in joined


def test_select_best_candidate_llm_clamps_out_of_range_index(monkeypatch):
    class _FakeParsedMessage:
        parsed_output = RerankVerdict(best_index=5, reasoning="hallucinated index")

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = select_best_candidate_llm("q", ["a", "b"])
    assert verdict.best_index == 0


@pytest.mark.slow
def test_select_best_candidate_llm_real_call():
    try:
        verdict = select_best_candidate_llm(
            "what is the capital of France",
            ["Paris has many museums and a famous tower.", "The capital of France is Paris."],
        )
    except anthropic.AnthropicError as exc:
        pytest.skip(f"Anthropic API not configured: {exc}")
    assert verdict.best_index in (0, 1)
    assert verdict.reasoning


def test_synthesis_verdict_model_shape():
    verdict = SynthesisVerdict(
        answerable=True,
        answer_text="Founded in 1998, the company went public in 2004.",
        source_indices=[0, 2],
        reasoning="Founding year is in passage 0, IPO year is in passage 2.",
    )
    assert verdict.answerable is True
    assert verdict.source_indices == [0, 2]


def test_synthesize_answer_llm_calls_sdk_with_expected_shape(monkeypatch):
    captured = {}

    class _FakeParsedMessage:
        parsed_output = SynthesisVerdict(
            answerable=True,
            answer_text="The company was founded in 1998 and went public in 2004.",
            source_indices=[0, 1],
            reasoning="Combines the founding year from passage 0 and the IPO year from passage 1.",
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = synthesize_answer_llm(
        "when was the company founded and when did it ipo",
        ["The company was founded in 1998.", "It went public via IPO in 2004."],
    )

    assert verdict.answerable is True
    assert verdict.source_indices == [0, 1]
    assert captured["model"] == "claude-sonnet-5"
    assert captured["output_format"] is SynthesisVerdict
    joined = " ".join(m["content"] for m in captured["messages"])
    assert "when was the company founded and when did it ipo" in joined
    assert "[0] The company was founded in 1998." in joined
    assert "[1] It went public via IPO in 2004." in joined


def test_synthesize_answer_llm_clears_answer_when_not_answerable(monkeypatch):
    # Never hallucinate: if the model says the evidence is insufficient, the
    # answer_text/source_indices need to be cleared in code, not merely
    # trusted from the prompt.
    class _FakeParsedMessage:
        parsed_output = SynthesisVerdict(
            answerable=False,
            answer_text="ignored — should be cleared",
            source_indices=[0],
            reasoning="None of the passages mention the requested date.",
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = synthesize_answer_llm("what date did X happen", ["passage a", "passage b"])
    assert verdict.answerable is False
    assert verdict.answer_text == ""
    assert verdict.source_indices == []


def test_synthesize_answer_llm_filters_out_of_range_indices(monkeypatch):
    class _FakeParsedMessage:
        parsed_output = SynthesisVerdict(
            answerable=True, answer_text="answer", source_indices=[0, 5, 1], reasoning="r"
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = synthesize_answer_llm("q", ["a", "b"])
    assert verdict.source_indices == [0, 1]


@pytest.mark.slow
def test_synthesize_answer_llm_real_call_recovers_multi_passage_answer():
    # Adversarial case: the answer requires information from two passages;
    # a single passage can't answer it alone.
    try:
        verdict = synthesize_answer_llm(
            "when was the company founded and when did it go public",
            [
                "Acme Corp was founded in 1998 by two engineers in a garage.",
                "Acme Corp completed its initial public offering in 2004.",
            ],
        )
    except anthropic.AnthropicError as exc:
        pytest.skip(f"Anthropic API not configured: {exc}")
    assert verdict.answerable is True
    assert "1998" in verdict.answer_text and "2004" in verdict.answer_text
