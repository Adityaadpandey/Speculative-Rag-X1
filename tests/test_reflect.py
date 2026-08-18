import anthropic
import pytest

from speculative_rag.reflect import (
    GeneratedAnswer,
    ReflectVerdict,
    RerankAndAnswer,
    RerankCandidate,
    RerankVerdict,
    SelectedSource,
    SynthesisVerdict,
    check_groundedness_llm,
    generate_answer_llm,
    rerank_and_generate_llm,
    rerank_passages_llm,
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
    assert captured["model"] == "claude-haiku-4-5-20251001"
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
    verdict = RerankVerdict(
        selected_sources=[SelectedSource(source_id="c1", relevance_score=0.9)],
        answerable=True,
        reasoning="Candidate 1 directly names the capital.",
    )
    assert verdict.selected_sources == [SelectedSource(source_id="c1", relevance_score=0.9)]
    assert verdict.answerable is True


def test_rerank_verdict_supports_no_relevant_sources():
    verdict = RerankVerdict(selected_sources=[], answerable=False, reasoning="None of these mention it.")
    assert verdict.selected_sources == []
    assert verdict.answerable is False


def test_rerank_passages_llm_calls_sdk_with_expected_shape(monkeypatch):
    from speculative_rag.reflect import _RerankParsed, _RerankSelection

    captured = {}

    class _FakeParsedMessage:
        parsed_output = _RerankParsed(
            selections=[_RerankSelection(index=1, relevance_score=0.87)],
            answerable=True,
            reasoning="Candidate 1 answers the question.",
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = rerank_passages_llm(
        "what is the capital of France",
        [
            RerankCandidate(source_id="museums", text="Paris has many museums."),
            RerankCandidate(source_id="capital", text="The capital of France is Paris."),
        ],
    )

    assert verdict.selected_sources == [SelectedSource(source_id="capital", relevance_score=0.87)]
    assert verdict.answerable is True
    assert captured["model"] == "claude-haiku-4-5-20251001"
    joined = " ".join(m["content"] for m in captured["messages"])
    assert "what is the capital of France" in joined
    assert "[0] Paris has many museums." in joined
    assert "[1] The capital of France is Paris." in joined


def test_rerank_passages_llm_filters_out_of_range_index(monkeypatch):
    from speculative_rag.reflect import _RerankParsed, _RerankSelection

    class _FakeParsedMessage:
        parsed_output = _RerankParsed(
            selections=[_RerankSelection(index=5, relevance_score=0.9)],
            answerable=True,
            reasoning="hallucinated index",
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = rerank_passages_llm("q", [RerankCandidate(source_id="a", text="a"), RerankCandidate(source_id="b", text="b")])
    assert verdict.selected_sources == []
    assert verdict.answerable is False


def test_rerank_passages_llm_never_claims_answerable_with_no_selections(monkeypatch):
    # Never trust answerable=true from the model alone if it selected nothing
    # -- that combination would let a downstream generator run on no evidence.
    from speculative_rag.reflect import _RerankParsed

    class _FakeParsedMessage:
        parsed_output = _RerankParsed(selections=[], answerable=True, reasoning="contradictory output")

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = rerank_passages_llm("q", [RerankCandidate(source_id="a", text="a")])
    assert verdict.answerable is False
    assert verdict.selected_sources == []


def test_rerank_passages_llm_returns_no_selection_when_nothing_relevant(monkeypatch):
    from speculative_rag.reflect import _RerankParsed

    class _FakeParsedMessage:
        parsed_output = _RerankParsed(
            selections=[], answerable=False, reasoning="None of the candidates mention this topic."
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = rerank_passages_llm(
        "what is the capital of France", [RerankCandidate(source_id="cats", text="Cats are popular pets.")]
    )
    assert verdict.selected_sources == []
    assert verdict.answerable is False


@pytest.mark.slow
def test_rerank_passages_llm_real_call():
    try:
        verdict = rerank_passages_llm(
            "what is the capital of France",
            [
                RerankCandidate(source_id="c0", text="Paris has many museums and a famous tower."),
                RerankCandidate(source_id="c1", text="The capital of France is Paris."),
            ],
        )
    except anthropic.AnthropicError as exc:
        pytest.skip(f"Anthropic API not configured: {exc}")
    assert verdict.reasoning


def test_generated_answer_model_shape():
    answer = GeneratedAnswer(answerable=True, answer_text="Paris is the capital of France.", reasoning="Stated directly.")
    assert answer.answerable is True
    assert "Paris" in answer.answer_text


def test_generate_answer_llm_calls_sdk_with_expected_shape(monkeypatch):
    captured = {}

    class _FakeParsedMessage:
        parsed_output = GeneratedAnswer(
            answerable=True,
            answer_text="The Manhattan Project successfully produced the atomic bomb, which helped end World War II.",
            reasoning="Directly stated in the evidence.",
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    answer = generate_answer_llm(
        "What was the immediate impact of the success of the Manhattan Project?",
        ["The Manhattan Project and its atomic bomb helped bring an end to World War II."],
    )

    assert answer.answerable is True
    assert "atomic bomb" in answer.answer_text
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["output_format"] is GeneratedAnswer
    joined = " ".join(m["content"] for m in captured["messages"])
    assert "What was the immediate impact of the success of the Manhattan Project?" in joined
    assert "The Manhattan Project and its atomic bomb helped bring an end to World War II." in joined
    assert "Do not add unsupported facts" in joined


def test_generate_answer_llm_clears_text_when_not_answerable(monkeypatch):
    class _FakeParsedMessage:
        parsed_output = GeneratedAnswer(
            answerable=False, answer_text="ignored — should be cleared", reasoning="Evidence doesn't cover this."
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    answer = generate_answer_llm("unrelated question", ["unrelated evidence"])
    assert answer.answerable is False
    assert answer.answer_text == ""


@pytest.mark.slow
def test_generate_answer_llm_real_call_answers_manhattan_project_question():
    try:
        answer = generate_answer_llm(
            "What was the immediate impact of the success of the Manhattan Project?",
            ["The Manhattan Project and its atomic bomb helped bring an end to World War II."],
        )
    except anthropic.AnthropicError as exc:
        pytest.skip(f"Anthropic API not configured: {exc}")
    assert answer.answerable is True
    assert "World War" in answer.answer_text or "war" in answer.answer_text.lower()


def test_rerank_and_generate_llm_calls_sdk_with_expected_shape(monkeypatch):
    from speculative_rag.reflect import _RerankAndGenerateParsed, _RerankAndGenerateSelection

    captured = {}

    class _FakeParsedMessage:
        parsed_output = _RerankAndGenerateParsed(
            selections=[_RerankAndGenerateSelection(index=1, relevance_score=0.9)],
            selection_found=True,
            answerable=True,
            answer_text="The capital of France is Paris.",
            reasoning="Candidate 1 names the capital directly.",
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = rerank_and_generate_llm(
        "what is the capital of France",
        [
            RerankCandidate(source_id="museums", text="Paris has many museums."),
            RerankCandidate(source_id="capital", text="The capital of France is Paris."),
        ],
    )

    assert verdict.selected_sources == [SelectedSource(source_id="capital", relevance_score=0.9)]
    assert verdict.selection_found is True
    assert verdict.answerable is True
    assert verdict.answer_text == "The capital of France is Paris."
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["output_format"] is _RerankAndGenerateParsed
    joined = " ".join(m["content"] for m in captured["messages"])
    assert "what is the capital of France" in joined
    assert "[0] Paris has many museums." in joined
    assert "[1] The capital of France is Paris." in joined


def test_rerank_and_generate_llm_clears_selection_and_answer_when_nothing_relevant(monkeypatch):
    from speculative_rag.reflect import _RerankAndGenerateParsed

    class _FakeParsedMessage:
        parsed_output = _RerankAndGenerateParsed(
            selections=[], selection_found=False, answerable=False, answer_text="",
            reasoning="None of the candidates mention this topic.",
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = rerank_and_generate_llm(
        "what is the capital of France", [RerankCandidate(source_id="cats", text="Cats are popular pets.")]
    )
    assert verdict.selected_sources == []
    assert verdict.selection_found is False
    assert verdict.answerable is False
    assert verdict.answer_text == ""


def test_rerank_and_generate_llm_never_answerable_without_a_selection(monkeypatch):
    # Same defensive contract as rerank_passages_llm/generate_answer_llm: never
    # trust answerable=true from the model if selection_found is false or no
    # sources were actually selected -- that combination would let a
    # downstream caller treat an unsupported guess as a real answer.
    from speculative_rag.reflect import _RerankAndGenerateParsed

    class _FakeParsedMessage:
        parsed_output = _RerankAndGenerateParsed(
            selections=[],
            selection_found=True,
            answerable=True,
            answer_text="a guess with no selected evidence",
            reasoning="contradictory output",
        )

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage()

    class _FakeAnthropic:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("speculative_rag.reflect.anthropic.Anthropic", _FakeAnthropic)

    verdict = rerank_and_generate_llm("q", [RerankCandidate(source_id="a", text="a")])
    assert verdict.selection_found is False
    assert verdict.answerable is False
    assert verdict.answer_text == ""


@pytest.mark.slow
def test_rerank_and_generate_llm_real_call():
    try:
        verdict = rerank_and_generate_llm(
            "what is the capital of France",
            [
                RerankCandidate(source_id="c0", text="Paris has many museums and a famous tower."),
                RerankCandidate(source_id="c1", text="The capital of France is Paris."),
            ],
        )
    except anthropic.AnthropicError as exc:
        pytest.skip(f"Anthropic API not configured: {exc}")
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
    assert captured["model"] == "claude-haiku-4-5-20251001"
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
