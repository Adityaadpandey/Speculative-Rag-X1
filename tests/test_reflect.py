import anthropic
import pytest

from speculative_rag.reflect import ReflectVerdict, check_groundedness_llm


def test_reflect_verdict_model_shape():
    verdict = ReflectVerdict(grounded=True, reasoning="The answer restates the source directly.")
    assert verdict.grounded is True
    assert "restates" in verdict.reasoning


def test_check_groundedness_llm_calls_sdk_with_expected_shape(monkeypatch):
    captured = {}

    class _FakeParsedMessage:
        parsed_output = ReflectVerdict(grounded=False, reasoning="Introduces an unsupported claim.")

    class _FakeMessages:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = check_groundedness_llm("Paris is in France.", "Paris, France's capital, sits on the Seine.")

    assert verdict.grounded is False
    assert captured["model"] == "claude-sonnet-5"
    assert captured["output_format"] is ReflectVerdict
    joined = " ".join(m["content"] for m in captured["messages"])
    assert "Paris is in France." in joined
    assert "Paris, France's capital, sits on the Seine." in joined


@pytest.mark.slow
def test_check_groundedness_llm_real_call():
    try:
        verdict = check_groundedness_llm(
            "Paris is the capital of France.",
            "France's capital city is Paris, located on the Seine river.",
        )
    except anthropic.AnthropicError as exc:
        pytest.skip(f"Anthropic API not configured: {exc}")
    assert isinstance(verdict.grounded, bool)
    assert verdict.reasoning
