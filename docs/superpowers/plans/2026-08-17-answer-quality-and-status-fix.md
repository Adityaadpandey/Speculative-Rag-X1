# Answer-Quality and Status-Logic Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the pipeline from marking answers `ANSWERED` when they are merely grounded-but-not-answering, by separating faithfulness/answerability/completeness/relevance into independent verdicts, fixing the extractive generator's full-passage-dump fallback, and adding an async multi-passage LLM synthesis path for when a single passage can't answer the query.

**Architecture:** The hot path (`pipeline.run_pipeline`) stays extractive, model-free, and fast (~200ms) — it only gets a targeted fix to stop treating "no sentence overlaps the query" as if the whole passage were still a valid answer. All LLM judgment (whether the answer *actually* answers the query, whether it's complete/relevant, and multi-passage synthesis) stays async and non-hot-path, exactly like the existing `reflect.py` groundedness/rerank calls. A new pure `answer_status.combine_verdict()` function is the single place that turns "hot-path outcome + optional async verdict" into the four-value status the user actually sees — `ANSWERED` never fires until that verdict has resolved and passed all four dimensions.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, `anthropic` SDK (`claude-sonnet-5`, structured output via `messages.parse`), Qdrant, pytest.

## Global Constraints

- No LLM call may be added to the synchronous path inside `run_pipeline` — that function's total latency budget is ~200ms (spec: embed ~218ms, retrieve ~3ms, generate <1ms). All new LLM calls (`reflect.py`) are invoked from `webapp/main.py` request handlers or eval scripts, never from `pipeline.py`.
- `RagOutcome` (in `pipeline_schemas.py`) is the hot-path's own vocabulary and keeps its existing 5 values unchanged — do not repurpose it as a correctness verdict. The new `FinalAnswerStatus` (4 values: `ANSWERED`, `NOT_ANSWERED`, `INSUFFICIENT_EVIDENCE`, `UNGROUNDED`) is the user-facing status and is computed by `answer_status.combine_verdict`.
- Never invent an answer from outside the retrieved evidence. `SynthesisVerdict.answerable=False` must force `answer_text=""` and `source_indices=[]` in code (not just via prompt wording) — this is a hard invariant, test it directly.
- All existing tests must keep passing; where a signature changes (`select_answer_span` now returns `str | None`, `ReflectVerdict` gains two required fields), update every call/construction site in the same task, not later.

---

### Task 1: Stop `select_answer_span` from dumping the whole passage as the answer

The exact bug from `results/answer_quality_en_native.md`: when no sentence in the passage shares any token with the query, the function currently returns the entire passage text, which then sails through the (trivially ~1.0) lexical groundedness check as if it were a real answer. Change the no-overlap fallback to signal "no answer here" instead.

**Files:**
- Modify: `src/speculative_rag/generation.py`
- Test: `tests/test_generation.py`

**Interfaces:**
- Produces: `select_answer_span(query_text: str, passage_text: str) -> str | None` — `None` means "no sentence in a multi-sentence passage overlaps the query at all." Single-sentence passages still return that sentence unconditionally (nothing better exists to fall back to). Task 2 consumes this `None`.

- [ ] **Step 1: Update the existing no-overlap test to expect `None`, and add the "many irrelevant sentences" regression test**

Replace the existing `test_select_span_no_overlap_falls_back_to_whole_passage` test and add a new one, in `tests/test_generation.py`:

```python
from speculative_rag.generation import select_answer_span


def test_select_span_picks_sentence_overlapping_query():
    passage = "Cats are mammals. The Eiffel Tower is in Paris. Dogs bark loudly."
    span = select_answer_span("where is the Eiffel Tower", passage)
    assert "Eiffel Tower" in span
    assert "Cats" not in span


def test_select_span_single_sentence_returns_whole_passage():
    passage = "A corporation is a legal entity"
    assert select_answer_span("what is a corporation", passage) == "A corporation is a legal entity"


def test_select_span_no_overlap_returns_none_instead_of_dumping_passage():
    # This is the exact failure mode from results/answer_quality_en_native.md:
    # a topically-plausible passage with zero query-token overlap used to be
    # returned WHOLE, then scored groundedness=1.0 trivially (the "answer"
    # *is* the source by construction) and marked ANSWERED. It must now
    # signal "no answer here" instead of masquerading as one.
    passage = "Cats are mammals. Dogs bark loudly."
    span = select_answer_span("quantum chromodynamics theory", passage)
    assert span is None


def test_select_span_picks_only_relevant_sentence_among_many_irrelevant():
    # Adversarial case 2 (spec): passage contains the answer plus 10
    # irrelevant sentences — the answer must contain only the relevant part,
    # not a dump of everything.
    passage = (
        "The weather today is sunny. Traffic was light this morning. "
        "Coffee prices rose last quarter. "
        "The immediate impact of the test was a bright flash visible for miles. "
        "Birds migrate south in winter. The stock market closed higher. "
        "Local schools are on break this week. Many tourists visited the museum. "
        "The bridge was repainted recently. A new cafe opened downtown. "
        "Rainfall was below average this year."
    )
    span = select_answer_span("what was the immediate impact of the test", passage)
    assert span == "The immediate impact of the test was a bright flash visible for miles."
```

- [ ] **Step 2: Run tests to verify the no-overlap test fails**

Run: `uv run pytest tests/test_generation.py -v`
Expected: `test_select_span_no_overlap_returns_none_instead_of_dumping_passage` FAILS (current code returns the whole passage string, not `None`). The other three tests, including the new "many irrelevant sentences" one, already PASS against the current implementation — that's expected; `max(sentences, key=overlap)` already does the right thing when overlap is nonzero for at least one sentence. Only the true-zero-overlap fallback is broken.

- [ ] **Step 3: Implement the fix**

In `src/speculative_rag/generation.py`, replace the function body:

```python
def select_answer_span(query_text: str, passage_text: str) -> str | None:
    """Return the sentence in the passage with the most token overlap with the
    query. Falls back to the whole passage only when it is a single sentence
    (there is nothing better to return). When there are multiple sentences
    but none shares any token with the query, that is a signal the passage
    does not contain the answer — return None rather than dumping the full
    passage as if it were the answer."""
    sentences = split_sentences(passage_text)
    if len(sentences) <= 1:
        return passage_text.strip()

    query_tokens = tokenize(query_text)

    def overlap(sentence: str) -> int:
        return len(tokenize(sentence) & query_tokens)

    best = max(sentences, key=overlap)
    if overlap(best) == 0:
        return None
    return best.strip()
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `uv run pytest tests/test_generation.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/generation.py tests/test_generation.py
git commit -m "fix(generation): return None instead of dumping the full passage when no sentence answers the query"
```

---

### Task 2: Wire the `None` span into an early `INSUFFICIENT_EVIDENCE` exit in the pipeline

**Files:**
- Modify: `src/speculative_rag/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `select_answer_span(query_text, passage_text) -> str | None` from Task 1.
- Produces: no change to `PipelineResult`/`RagOutcome` shape — reuses the existing `RagOutcome.INSUFFICIENT_EVIDENCE` value, just adds a new, earlier place it can fire from.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
def test_insufficient_evidence_when_no_sentence_overlaps_query_at_all():
    # Mirrors the exact bug pattern from results/answer_quality_en_native.md:
    # a topically-similar (high embedding score), on-topic passage that
    # contains no sentence answering the query used to be returned WHOLE as
    # if it were the answer. It must now short-circuit before generation
    # ever produces a candidate "answer" to grade.
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored(
                    "Manhattan Project scientists gathered at Los Alamos. "
                    "They worked for several years on nuclear physics research.",
                    0.85,
                )
            ]
        )
    )
    result = _run("immediate impact of successful test", retriever)

    assert result.outcome == RagOutcome.INSUFFICIENT_EVIDENCE
    assert result.answer is None
    assert result.refusal_reason is not None
    assert "no sentence" in result.refusal_reason
    assert [t.stage for t in result.timings] == [
        "guardrail_unsafe",
        "embed",
        "retrieve",
        "guardrail_offtopic",
        "generate",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py::test_insufficient_evidence_when_no_sentence_overlaps_query_at_all -v`
Expected: FAILS. Currently `select_answer_span` still returns the full passage text for this case (Task 1 not yet applied when run standalone — but since Task 1 is already done by this point in execution order, the real expected failure here is that `pipeline.py` has no `None`-handling branch yet, so `groundedness_score(None, ...)` / `tokenize(None)` raises `TypeError`, or — since Task 1 already lands `None` — the pipeline currently crashes at `groundedness_score(span, ...)` because `span` is `None`). Confirm it fails (crash or wrong outcome), not that it silently passes.

- [ ] **Step 3: Implement the fix**

In `src/speculative_rag/pipeline.py`, insert a new early exit immediately after the `generate` stage, before the groundedness guardrail:

```python
    with timer.stage("generate"):
        span = select_answer_span(query_text, top.chunk.text)
    if span is None:
        return PipelineResult(
            outcome=RagOutcome.INSUFFICIENT_EVIDENCE,
            refusal_reason=(
                "no sentence in the retrieved passage shares vocabulary with the query text"
            ),
            top_score=top_score,
            timings=timer.timings,
        )

    with timer.stage("guardrail_groundedness"):
        ground = groundedness_score(span, top.chunk.text)
```

(This replaces the existing two lines `with timer.stage("generate"): span = select_answer_span(...)` immediately followed by `with timer.stage("guardrail_groundedness"):` — insert the `if span is None:` block between them, matching indentation of the surrounding function.)

- [ ] **Step 4: Run tests to verify all pass, including the full existing suite**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: all tests PASS, including the new one and the pre-existing `test_insufficient_evidence_when_extracted_span_lacks_expected_answer_type` (unaffected — its passage has nonzero overlap on the best sentence, so it still reaches the `guardrail_answer_type` gate as before).

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/pipeline.py tests/test_pipeline.py
git commit -m "fix(pipeline): exit INSUFFICIENT_EVIDENCE when no sentence in the passage answers the query"
```

---

### Task 3: Split the async LLM judge into four independent dimensions

`reflect.py` currently checks two of the four required dimensions (`grounded`, `answers_query`). Add `complete` (does the answer contain the required information) and `relevant` (is it focused on the query, not a context dump).

**Files:**
- Modify: `src/speculative_rag/reflect.py`
- Test: `tests/test_reflect.py`

**Interfaces:**
- Produces: `ReflectVerdict(grounded: bool, answers_query: bool, complete: bool, relevant: bool, reasoning: str)`. Task 4's `combine_verdict` consumes all four fields by name.

- [ ] **Step 1: Write the failing tests**

In `tests/test_reflect.py`, update the two existing tests that construct `ReflectVerdict` to include the two new required fields, and extend the real-call test's assertions:

```python
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
```

```python
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
    # ... (rest of the fake harness is unchanged) ...

    verdict = check_groundedness_llm(
        "what country is Paris in",
        "Paris is in France.",
        "Paris, France's capital, sits on the Seine.",
    )

    assert verdict.grounded is False
    assert verdict.complete is False
    assert verdict.relevant is True
    # ... (rest of assertions unchanged) ...
```

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reflect.py -v -m "not slow"`
Expected: FAIL with `pydantic.ValidationError` (missing required fields `complete`/`relevant` don't exist yet on `ReflectVerdict`, or the test file itself references fields that don't exist).

- [ ] **Step 3: Implement the fix**

In `src/speculative_rag/reflect.py`, replace `_PROMPT_TEMPLATE` and `ReflectVerdict`:

```python
_PROMPT_TEMPLATE = """You are reviewing an answer produced for a user's query, checking it against \
its source passage. Evaluate FOUR separate questions independently — do not let one verdict influence \
another. A verbatim, fully-supported excerpt from the source passage can still fail multiple of these.

User query:
{query_text}

Source passage:
{source_text}

Answer:
{answer_text}

1. grounded (faithfulness): Is every claim in the answer supported by the source passage, with no \
claims introduced that aren't present in the source passage?

2. answers_query (answerability): Does the answer directly and specifically answer the user's query — \
stating the specific fact, entity, date, number, or yes/no verdict the query is actually asking for? \
A verbatim excerpt that is merely on-topic but does NOT state the specific thing the query asks for \
must be scored answers_query=false, even though it may still be grounded=true.

3. complete (completeness): Does the answer contain the important information required to fully \
answer the query, or is it missing a necessary part of the answer?

4. relevant (relevance): Is the answer focused on the query, rather than mostly consisting of \
surrounding context, tangents, or information unrelated to what was asked?

These four verdicts are independent: an answer can be grounded but not answer the query, can answer \
the query but include an unsupported claim, can answer the query yet be incomplete, or any other \
combination — including all pass or all fail.

Give a brief one or two sentence reasoning covering all four verdicts."""


class ReflectVerdict(BaseModel):
    grounded: bool
    answers_query: bool
    complete: bool
    relevant: bool
    reasoning: str
```

The rest of `check_groundedness_llm` (signature, SDK call) is unchanged — only the prompt text and the model class change.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reflect.py -v -m "not slow"`
Expected: all PASS.

- [ ] **Step 5: Fix the remaining `ReflectVerdict` construction site and commit**

`tests/test_webapp.py` also constructs a `ReflectVerdict` directly (`fake_check_groundedness_llm`) — this is fixed in Task 6 alongside the webapp endpoint change, not here, since that test also needs `final_status` assertions added at the same time. For now, just commit this task's changes:

```bash
git add src/speculative_rag/reflect.py tests/test_reflect.py
git commit -m "feat(reflect): split the async LLM judge into four independent verdicts (grounded, answers_query, complete, relevant)"
```

---

### Task 4: `answer_status.combine_verdict` — the single place ANSWERED actually means answered

This is the core fix requested: a pure function that turns "hot-path outcome + optional async verdict" into the four-value status, with `ANSWERED` gated behind all four LLM dimensions passing.

**Files:**
- Create: `src/speculative_rag/answer_status.py`
- Test: `tests/test_answer_status.py`

**Interfaces:**
- Consumes: `RagOutcome` (from `pipeline_schemas.py`), `ReflectVerdict` (from `reflect.py`, Task 3).
- Produces: `FinalAnswerStatus` enum and `combine_verdict(outcome: RagOutcome, verdict: ReflectVerdict | None) -> tuple[FinalAnswerStatus | None, str]`. `status is None` means "hot path answered but the async verdict hasn't resolved yet — do not display as answered." Task 6 (webapp) and Task 8 (eval) both call this.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_answer_status.py`:

```python
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
    # Adversarial case 3 (spec): the answer copies the passage exactly
    # (grounded=1.0 in the old lexical sense) but never states what the
    # query asked for. This must NOT resolve to ANSWERED.
    verdict = _verdict(answers_query=False)
    status, reason = combine_verdict(RagOutcome.ANSWERED, verdict)
    assert status == FinalAnswerStatus.NOT_ANSWERED
    assert "does not answer" in reason


def test_case4_answers_query_but_has_unsupported_claim_is_ungrounded():
    # Adversarial case 4 (spec): answers the question correctly but adds one
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
    # Adversarial case 1 (spec): retrieved passage is on-topic but does not
    # contain the answer. Handled entirely in the hot path by Task 2; this
    # just confirms combine_verdict passes that outcome straight through.
    status, _ = combine_verdict(RagOutcome.INSUFFICIENT_EVIDENCE, None)
    assert status == FinalAnswerStatus.INSUFFICIENT_EVIDENCE


def test_ungrounded_refused_outcome_maps_to_ungrounded():
    status, _ = combine_verdict(RagOutcome.UNGROUNDED_REFUSED, None)
    assert status == FinalAnswerStatus.UNGROUNDED


def test_unsafe_input_refused_maps_to_not_answered():
    status, _ = combine_verdict(RagOutcome.UNSAFE_INPUT_REFUSED, None)
    assert status == FinalAnswerStatus.NOT_ANSWERED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_answer_status.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.answer_status'`.

- [ ] **Step 3: Implement `answer_status.py`**

Create `src/speculative_rag/answer_status.py`:

```python
"""Combines the fast, deterministic hot-path RagOutcome with the async,
four-dimension LLM ReflectVerdict into the single status actually shown to
an end user. Kept out of pipeline.py deliberately (the ~4s LLM groundedness
call must never enter the synchronous critical path) — this function is
pure and cheap, called by webapp/eval code once the async verdict resolves.

Core rule: RagOutcome.ANSWERED only means "the hot path produced a
candidate extractive answer that passed cheap heuristic gates." It is NOT a
correctness verdict. A grounded=1.0, verbatim-from-source answer that
doesn't state what the query asked for must resolve to NOT_ANSWERED here,
never ANSWERED."""

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
    ANSWERED and the async verdict hasn't resolved yet — callers must treat
    that as "pending," never display it as answered."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_answer_status.py -v`
Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/answer_status.py tests/test_answer_status.py
git commit -m "feat(answer_status): add FinalAnswerStatus and combine_verdict — the real answer-status decision logic"
```

---

### Task 5: Async multi-passage LLM synthesis

Following the exact non-hot-path pattern of `check_groundedness_llm`/`select_best_candidate_llm`: when a single passage can't answer the query, make one more LLM call across the top-k candidates asking it to synthesize (not copy) an answer only if the evidence supports one — never hallucinate.

**Files:**
- Modify: `src/speculative_rag/reflect.py`
- Test: `tests/test_reflect.py`

**Interfaces:**
- Produces: `SynthesisVerdict(answerable: bool, answer_text: str, source_indices: list[int], reasoning: str)` and `synthesize_answer_llm(query_text: str, candidate_texts: list[str]) -> SynthesisVerdict`. Task 6 (webapp `/api/synthesize`) and Task 9 (eval script) both call this.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_reflect.py`:

```python
from speculative_rag.reflect import SynthesisVerdict, synthesize_answer_llm


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
    # Spec requirement: never hallucinate. If the model says the evidence is
    # insufficient, the answer_text/source_indices must be cleared in code —
    # not merely trusted from the prompt.
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
    # Adversarial case 5 (spec): the answer requires information from two
    # passages; a single passage cannot answer it alone.
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reflect.py -v -m "not slow"`
Expected: FAIL with `ImportError: cannot import name 'SynthesisVerdict'`.

- [ ] **Step 3: Implement `synthesize_answer_llm`**

Append to `src/speculative_rag/reflect.py`:

```python
# Async multi-passage synthesis: when extractive top-1 selection fails
# (select_answer_span returned None, or the async reflect verdict says the
# top-1 answer is incomplete/doesn't answer the query), a single passage may
# still not contain the full answer even though the top-k collectively does.
# This makes ONE more LLM call across up to k candidate passages, asking it
# to synthesize — not copy — a concise answer ONLY if the evidence actually
# supports one, and to say so explicitly (never invent from world knowledge)
# when it doesn't. Non-hot-path, same pattern as the two functions above.
_SYNTHESIS_PROMPT_TEMPLATE = """You are answering a user's question using ONLY the evidence in the \
candidate passages below. Do not use outside knowledge.

Question:
{query_text}

Candidate passages:
{candidates_block}

First decide: do these passages, taken together, contain enough information to answer the question?

If yes: write a concise answer (one or two sentences) that states the specific fact the question asks \
for. Synthesize from the relevant parts of the passages — do not copy an entire passage verbatim, and \
do not include information irrelevant to the question. List the 0-based indices of the passages you \
actually used.

If no: do not guess or invent an answer from general knowledge. Set answerable=false, leave \
answer_text empty, and briefly explain what information is missing."""


class SynthesisVerdict(BaseModel):
    answerable: bool
    answer_text: str
    source_indices: list[int]
    reasoning: str


def synthesize_answer_llm(query_text: str, candidate_texts: list[str]) -> SynthesisVerdict:
    client = anthropic.Anthropic()
    candidates_block = "\n".join(f"[{i}] {text}" for i, text in enumerate(candidate_texts))
    response = client.messages.parse(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": _SYNTHESIS_PROMPT_TEMPLATE.format(
                    query_text=query_text, candidates_block=candidates_block
                ),
            }
        ],
        output_format=SynthesisVerdict,
    )
    verdict = response.parsed_output
    assert verdict is not None
    verdict.source_indices = [i for i in verdict.source_indices if 0 <= i < len(candidate_texts)]
    if not verdict.answerable:
        verdict.answer_text = ""
        verdict.source_indices = []
    return verdict
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reflect.py -v -m "not slow"`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/reflect.py tests/test_reflect.py
git commit -m "feat(reflect): add async multi-passage LLM synthesis with a hard never-hallucinate invariant"
```

---

### Task 6: Wire the four-dimension verdict and synthesis into the webapp

**Files:**
- Modify: `webapp/main.py`
- Test: `tests/test_webapp.py`

**Interfaces:**
- Consumes: `combine_verdict` (Task 4), `synthesize_answer_llm` (Task 5), extended `ReflectVerdict` (Task 3).
- Produces: `/api/reflect` response gains `complete`, `relevant`, `final_status`, `reason`. New `/api/synthesize` endpoint.

- [ ] **Step 1: Write the failing tests**

In `tests/test_webapp.py`, update `test_post_reflect_returns_grounded_verdict_shape`'s fake and assertions, and add new tests for `/api/synthesize`:

```python
def test_post_reflect_returns_grounded_verdict_shape(monkeypatch):
    # ... existing setup unchanged up through defining fake_check_groundedness_llm ...

    captured = {}

    def fake_check_groundedness_llm(query_text, answer_text, source_text):
        captured["query_text"] = query_text
        captured["answer_text"] = answer_text
        captured["source_text"] = source_text
        return ReflectVerdict(
            grounded=True,
            answers_query=True,
            complete=True,
            relevant=True,
            reasoning="Matches the source directly.",
        )

    monkeypatch.setattr(webapp_main, "check_groundedness_llm", fake_check_groundedness_llm)

    # ... existing request POST unchanged ...

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert data["answers_query"] is True
    assert data["complete"] is True
    assert data["relevant"] is True
    assert data["final_status"] == "answered"
    assert data["reasoning"] == "Matches the source directly."
    assert data["reflect_latency_ms"] >= 0.0
    assert captured["query_text"] == "what is a corporation"
    assert captured["answer_text"] == "A corporation is a company."
    assert captured["source_text"] == "A corporation is a company."


def test_post_reflect_ungrounded_verdict_maps_to_not_answered_family(monkeypatch):
    monkeypatch.setattr(webapp_main, "_EMBEDDER", FakeEmbedder())
    retriever = FakeRetriever(
        RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9, pid="p1")])
    )
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    def fake_check_groundedness_llm(query_text, answer_text, source_text):
        return ReflectVerdict(
            grounded=True,
            answers_query=False,
            complete=False,
            relevant=True,
            reasoning="On-topic but never states the specific fact asked for.",
        )

    monkeypatch.setattr(webapp_main, "check_groundedness_llm", fake_check_groundedness_llm)

    response = client.post(
        "/api/reflect",
        json={
            "query_text": "what is a corporation",
            "language": "hi",
            "strategy": "native",
            "answer_text": "A corporation is a company.",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["final_status"] == "not_answered"
    assert "does not answer" in data["reason"]


def test_post_synthesize_returns_answer_when_answerable(monkeypatch):
    from speculative_rag.reflect import SynthesisVerdict

    monkeypatch.setattr(webapp_main, "_EMBEDDER", FakeEmbedder())
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored("Acme Corp was founded in 1998.", 0.9, pid="p_founded"),
                _scored("Acme Corp went public in 2004.", 0.85, pid="p_ipo"),
            ]
        )
    )
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    def fake_synthesize_answer_llm(query_text, candidate_texts):
        return SynthesisVerdict(
            answerable=True,
            answer_text="Acme Corp was founded in 1998 and went public in 2004.",
            source_indices=[0, 1],
            reasoning="Combines founding year and IPO year.",
        )

    monkeypatch.setattr(webapp_main, "synthesize_answer_llm", fake_synthesize_answer_llm)

    response = client.post(
        "/api/synthesize",
        json={"query_text": "when was Acme founded and when did it ipo", "language": "hi", "strategy": "native"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answerable"] is True
    assert data["answer_text"] == "Acme Corp was founded in 1998 and went public in 2004."
    assert data["sources"] == [
        {"source_passage_id": "p_founded", "retrieval_score": 0.9},
        {"source_passage_id": "p_ipo", "retrieval_score": 0.85},
    ]
    assert data["synthesis_latency_ms"] >= 0.0


def test_post_synthesize_returns_not_answerable_shape(monkeypatch):
    from speculative_rag.reflect import SynthesisVerdict

    monkeypatch.setattr(webapp_main, "_EMBEDDER", FakeEmbedder())
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("unrelated text", 0.9, pid="p1")]))
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    def fake_synthesize_answer_llm(query_text, candidate_texts):
        return SynthesisVerdict(answerable=False, answer_text="", source_indices=[], reasoning="Missing.")

    monkeypatch.setattr(webapp_main, "synthesize_answer_llm", fake_synthesize_answer_llm)

    response = client.post(
        "/api/synthesize",
        json={"query_text": "x", "language": "hi", "strategy": "native"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answerable"] is False
    assert data["answer_text"] == ""
    assert data["sources"] == []


def test_post_synthesize_unknown_collection_returns_404(monkeypatch):
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {})
    response = client.post(
        "/api/synthesize",
        json={"query_text": "x", "language": "hi", "strategy": "native"},
    )
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_webapp.py -v`
Expected: FAIL — `test_post_reflect_returns_grounded_verdict_shape` fails on the new `data["complete"]`/`final_status` assertions (or on `ReflectVerdict` missing-field validation), and the three new `/api/synthesize` tests fail with 404s (endpoint doesn't exist) or `AttributeError` on `webapp_main.synthesize_answer_llm`.

- [ ] **Step 3: Implement the webapp changes**

In `webapp/main.py`, update imports:

```python
from speculative_rag.answer_status import combine_verdict
from speculative_rag.config import BCP47_LANGUAGE_CODES, LANGUAGE_SOURCES, SETTINGS
from speculative_rag.embedding import Embedder
from speculative_rag.generation import select_answer_span
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.reflect import check_groundedness_llm, select_best_candidate_llm, synthesize_answer_llm
from speculative_rag.retrieval import Retriever
from speculative_rag.stt import SarvamAPIKeyMissingError, transcribe_audio
```

Replace the body of `post_reflect`:

```python
@app.post("/api/reflect", response_model=None)
async def post_reflect(body: ReflectRequest) -> JSONResponse | dict:
    collection_name = f"{body.language}_{body.strategy}"
    retriever = _RETRIEVERS.get(collection_name)
    if retriever is None:
        return JSONResponse(status_code=404, content={"error": f"No such collection '{collection_name}'"})

    query_vector = _EMBEDDER.embed_query(body.query_text)
    retrieval = retriever.search(query_vector, k=1)
    source_text = retrieval.top.chunk.text if retrieval.top is not None else ""

    reflect_start = time.perf_counter()
    verdict = check_groundedness_llm(body.query_text, body.answer_text, source_text)
    reflect_ms = (time.perf_counter() - reflect_start) * 1000.0

    final_status, reason = combine_verdict(RagOutcome.ANSWERED, verdict)

    return {
        "grounded": verdict.grounded,
        "answers_query": verdict.answers_query,
        "complete": verdict.complete,
        "relevant": verdict.relevant,
        "final_status": final_status.value if final_status is not None else None,
        "reason": reason,
        "reasoning": verdict.reasoning,
        "reflect_latency_ms": reflect_ms,
    }
```

Append a new endpoint after `post_rerank`:

```python
class SynthesizeRequest(BaseModel):
    query_text: str
    language: str
    strategy: str


@app.post("/api/synthesize", response_model=None)
async def post_synthesize(body: SynthesizeRequest) -> JSONResponse | dict:
    collection_name = f"{body.language}_{body.strategy}"
    retriever = _RETRIEVERS.get(collection_name)
    if retriever is None:
        return JSONResponse(status_code=404, content={"error": f"No such collection '{collection_name}'"})

    query_vector = _EMBEDDER.embed_query(body.query_text)
    retrieval = retriever.search(query_vector, k=SETTINGS.answer_top_k)
    candidates = retrieval.scored_chunks
    if not candidates:
        return JSONResponse(status_code=404, content={"error": "no candidates retrieved for synthesis"})

    synth_start = time.perf_counter()
    verdict = synthesize_answer_llm(body.query_text, [c.chunk.text for c in candidates])
    synth_ms = (time.perf_counter() - synth_start) * 1000.0

    sources = [
        {"source_passage_id": candidates[i].chunk.source_passage_id, "retrieval_score": candidates[i].score}
        for i in verdict.source_indices
    ]

    return {
        "answerable": verdict.answerable,
        "answer_text": verdict.answer_text,
        "sources": sources,
        "reasoning": verdict.reasoning,
        "synthesis_latency_ms": synth_ms,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_webapp.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add webapp/main.py tests/test_webapp.py
git commit -m "feat(webapp): compute final_status via combine_verdict, add /api/synthesize"
```

---

### Task 7: Surface the real status and synthesis fallback in the UI

**Files:**
- Modify: `webapp/static/index.html`
- Modify: `webapp/static/app.js`

- [ ] **Step 1: Add a synthesis panel and a final-status badge to the review panel**

In `webapp/static/index.html`, inside the existing `<section id="review-panel" hidden>` block, add a badge line right after the opening heading/description (before `#review-status`):

```html
      <p class="meta">final status: <strong id="review-final-status">pending…</strong> — <span id="review-final-reason"></span></p>
```

Add a new section immediately after the existing `<section id="rerank-panel" hidden>` block (before `<p id="error-banner" ...>`):

```html
    <section id="synthesis-panel" hidden>
      <h2>Multi-passage synthesis</h2>
      <p class="meta">Async fallback when a single passage can't answer the query — synthesizes across the top-k candidates instead of guessing, or explicitly reports insufficient evidence.</p>
      <div id="synthesis-status">checking multiple passages...</div>
      <div id="synthesis-result" hidden>
        <p id="synthesis-answer"></p>
        <p class="meta">
          answerable: <span id="synthesis-answerable"></span> ·
          sources: <span id="synthesis-sources"></span> ·
          latency: <span id="synthesis-latency-ms"></span> ms
        </p>
        <p id="synthesis-reasoning"></p>
      </div>
    </section>
```

- [ ] **Step 2: Wire up the JS**

In `webapp/static/app.js`, add new element references near the existing `rerankPanel` block:

```javascript
const reviewFinalStatus = document.getElementById("review-final-status");
const reviewFinalReason = document.getElementById("review-final-reason");
const synthesisPanel = document.getElementById("synthesis-panel");
const synthesisStatus = document.getElementById("synthesis-status");
const synthesisResult = document.getElementById("synthesis-result");
const synthesisAnswer = document.getElementById("synthesis-answer");
const synthesisAnswerable = document.getElementById("synthesis-answerable");
const synthesisSources = document.getElementById("synthesis-sources");
const synthesisLatencyMs = document.getElementById("synthesis-latency-ms");
const synthesisReasoning = document.getElementById("synthesis-reasoning");
```

In `resetPanels()`, add:

```javascript
  synthesisPanel.hidden = true;
  synthesisResult.hidden = true;
  synthesisStatus.hidden = false;
```

Add a `startSynthesis` function (mirroring `startRerank`) after `startRerank`:

```javascript
async function startSynthesis(queryText) {
  synthesisPanel.hidden = false;
  synthesisStatus.hidden = false;
  synthesisResult.hidden = true;

  if (!state.anthropicKeyConfigured) {
    synthesisStatus.textContent = "ANTHROPIC_API_KEY not set — skipping synthesis";
    return;
  }

  synthesisStatus.textContent = "checking multiple passages...";
  try {
    const response = await fetch("/api/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_text: queryText,
        language: languageSelect.value,
        strategy: strategySelect.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      synthesisStatus.textContent = `synthesis failed: ${data.error ?? "unknown error"}`;
      return;
    }
    synthesisStatus.hidden = true;
    synthesisResult.hidden = false;
    synthesisAnswerable.textContent = data.answerable;
    synthesisAnswer.textContent = data.answerable
      ? data.answer_text
      : "The retrieved evidence does not contain enough information to answer this fully.";
    synthesisSources.textContent = data.sources.map((s) => s.source_passage_id).join(", ") || "none";
    synthesisLatencyMs.textContent = data.synthesis_latency_ms.toFixed(2);
    synthesisReasoning.textContent = data.reasoning;
  } catch (err) {
    synthesisStatus.textContent = `synthesis failed: ${err.message}`;
  }
}
```

Update `startReview` to render the final-status badge once the reflect response resolves — add at the end of its success branch (after the existing `reviewReasoning.textContent = ...` line, before the closing of the `try` block):

```javascript
    reviewFinalStatus.textContent = data.final_status ?? "pending";
    reviewFinalReason.textContent = data.reason ?? "";
    if (data.final_status && data.final_status !== "answered") {
      startSynthesis(queryText);
    }
```

Update `runQuery`'s outcome handling so INSUFFICIENT_EVIDENCE (Task 2's new early exit) also triggers synthesis directly, since there's no extractive answer to send to `/api/reflect` at all:

```javascript
  renderResult(data);
  if (data.outcome === "answered") {
    startReview(queryText, data.answer.text);
    startRerank(queryText);
  } else if (data.outcome === "insufficient_evidence") {
    startSynthesis(queryText);
  }
```

- [ ] **Step 3: Manual verification**

Run: `uv run python -m scripts.serve_ui` and open the local URL. Submit a query that previously would have dumped a whole passage (e.g. an `en_native` DESCRIPTION query from `results/answer_quality_en_native.md` such as "what structure is found on the rough er") and confirm:
- If the hot path now returns `insufficient_evidence`, the synthesis panel runs automatically and either produces a real synthesized answer or an explicit "does not contain enough information" message — never a silent full-passage dump.
- If the hot path returns `answered`, the review panel's final-status badge shows `not_answered`/`ungrounded`/`answered` (not just the raw `grounded` flag), and non-`answered` verdicts trigger the synthesis panel automatically.

- [ ] **Step 4: Commit**

```bash
git add webapp/static/index.html webapp/static/app.js
git commit -m "feat(webapp-ui): surface final_status and an automatic multi-passage synthesis fallback"
```

---

### Task 8: Fix the answer-quality eval report to stop hiding failures behind `ANSWERED`

`scripts/run_answer_quality_eval.py` currently reports "300/300 answered" with F1 as a footnote — exactly the bug pattern the whole plan exists to fix, in the reporting layer. Add a sampled async-verification pass (same cost-bounding pattern as `run_llm_rerank_eval.py`) that reports the real `FinalAnswerStatus` breakdown among "answered" rows.

**Files:**
- Modify: `scripts/run_answer_quality_eval.py`
- No new automated test (this is a reporting script over live data/API calls, consistent with `run_llm_rerank_eval.py` having none); verify by running it once against an existing collection.

**Interfaces:**
- Consumes: `combine_verdict` (Task 4), `check_groundedness_llm` (Task 3).

- [ ] **Step 1: Implement the change**

In `scripts/run_answer_quality_eval.py`, add imports:

```python
from speculative_rag.answer_status import combine_verdict
from speculative_rag.reflect import check_groundedness_llm
```

The main loop itself needs no changes — it does not need to carry `source_text` forward. The verification pass added below re-embeds each sampled query and re-retrieves its top-1 chunk text directly, which is simpler than threading `source_text` through `rows` and correct regardless of chunking strategy (a stored `source_passage_id` isn't guaranteed to map back to a single chunk's exact text for non-`native` strategies).

Add the verification pass after the main loop, before building `report_df`:

```python
    report_df = pd.DataFrame(rows)
    answered = report_df[report_df["outcome"] == "answered"]

    verify_n = min(SETTINGS.reflect_latency_sample_size, len(answered))
    status_counts: dict[str, int] = {}
    if verify_n > 0:
        print(f"Running async LLM verification on {verify_n} ANSWERED rows...")
        verify_sample = answered.sample(n=verify_n, random_state=SETTINGS.sample_seed)
        for _, r in verify_sample.iterrows():
            query_vector = embedder.embed_query(r["query"])
            top = retriever.search(query_vector, k=1).top
            source_text = top.chunk.text if top is not None else ""
            try:
                verdict = check_groundedness_llm(r["query"], r["answer"], source_text)
            except anthropic.AnthropicError as exc:
                print(f"Anthropic API not configured, skipping verification: {exc}")
                status_counts = {}
                break
            status, _ = combine_verdict(RagOutcome.ANSWERED, verdict)
            status_counts[status.value] = status_counts.get(status.value, 0) + 1
```

Add `import anthropic` to the top of `scripts/run_answer_quality_eval.py` alongside the other imports.

Add a new report section (after the existing "Answer quality (among ANSWERED queries)" section, still inside the `if len(answered) > 0:` block), guarded so it's skipped entirely if verification never ran (e.g. no API key configured):

```python
        if status_counts:
            lines.append(
                f"\n## Real answer status among ANSWERED rows "
                f"({verify_n}-row LLM-verified sample — this is what 'answered' actually means)\n"
            )
            lines.append(
                "The pipeline's ANSWERED outcome only means a candidate extractive answer passed cheap "
                "heuristic gates — it is not a correctness verdict. This table applies the async four-"
                "dimension LLM judge (faithfulness/answerability/completeness/relevance) to a sample of "
                "those rows and reports what they actually resolve to.\n"
            )
            lines.append("| final status | count | % of sample |\n|---|---|---|\n")
            for status, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {status} | {count} | {count / verify_n * 100:.1f}% |\n")
```

- [ ] **Step 2: Run it against an existing collection to verify**

Run: `uv run python -m scripts.run_answer_quality_eval en native`
Expected: script completes, prints "Running async LLM verification on N ANSWERED rows...". Confirm the new "Real answer status" table appears in the printed output and in `results/answer_quality_en_native.md`, and that its `answered` percentage is visibly lower than the "300/300 answered" outcome-breakdown count above it — that gap is the bug this whole plan fixes, now measured instead of hidden. If `ANTHROPIC_API_KEY` isn't configured, confirm the script still completes and prints the skip message instead of crashing.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_answer_quality_eval.py results/answer_quality_en_native.md results/answer_quality_hi_native.md results/answer_quality_ta_native.md
git commit -m "feat(eval): report real LLM-verified FinalAnswerStatus breakdown instead of trusting ANSWERED alone"
```

---

### Task 9: New eval script — does multi-passage synthesis recover `INSUFFICIENT_EVIDENCE` queries?

**Files:**
- Create: `scripts/run_synthesis_eval.py`
- No automated test (mirrors the untested reporting-script pattern of `run_llm_rerank_eval.py`); verify by running it once.

**Interfaces:**
- Consumes: `synthesize_answer_llm` (Task 5), `run_pipeline` (existing), `RagOutcome.INSUFFICIENT_EVIDENCE`.

- [ ] **Step 1: Implement the script**

Create `scripts/run_synthesis_eval.py`, closely mirroring `scripts/run_llm_rerank_eval.py`'s structure:

```python
"""Does async multi-passage LLM synthesis recover queries the hot path marks
INSUFFICIENT_EVIDENCE (no single retrieved sentence answers the query)?
Runs the real pipeline over a sample, and for every INSUFFICIENT_EVIDENCE
row, additionally calls speculative_rag.reflect.synthesize_answer_llm over
the full top-k candidate set to see whether the evidence collectively
answers the query even though no single passage does.

Makes one real Anthropic API call per INSUFFICIENT_EVIDENCE row, so this
runs on a small sample (SETTINGS.reflect_latency_sample_size, default 30).

Usage:
    uv run python -m scripts.run_synthesis_eval [language] [strategy]
Defaults to hi/native. Requires ANTHROPIC_API_KEY and an existing collection.
"""

import sys
import time

import pandas as pd
from qdrant_client import QdrantClient

from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS
from speculative_rag.data import load_validation_sample
from speculative_rag.embedding import Embedder
from speculative_rag.latency import summarize
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.reflect import synthesize_answer_llm
from speculative_rag.retrieval import Retriever
from speculative_rag.text_utils import tokenize

_GOLD_ANSWER_FIELD = {
    "Translated_passages": "Answer",
    "English_passages": "Eng_Answer",
}


def word_overlap_f1(predicted: str, gold: str) -> float:
    pred_tokens = tokenize(predicted)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def run_eval(language: str, strategy: str) -> None:
    source = next(s for s in LANGUAGE_SOURCES if s.language == language)
    gold_field = _GOLD_ANSWER_FIELD[source.text_field]
    collection_name = f"{language}_{strategy}"

    print(f"Loading embedder and connecting to collection '{collection_name}'...")
    embedder = Embedder()
    client = QdrantClient(path=str(SETTINGS.qdrant_path))
    if not client.collection_exists(collection_name):
        print(f"No such collection '{collection_name}'. Run scripts.run_recall_eval first.")
        return
    retriever = Retriever(client, collection_name)

    sample_size = SETTINGS.reflect_latency_sample_size
    df = load_validation_sample(source.parquet_path, sample_size, SETTINGS.sample_seed)

    rows = []
    synthesis_latencies_ms: list[float] = []
    insufficient_count = 0
    for i, (_, row) in enumerate(df.iterrows()):
        query_text = row["query"] if source.text_field == "Translated_passages" else row["Eng_Query"]
        gold_answer = row[gold_field]

        result = run_pipeline(
            query_text,
            embedder,
            retriever,
            off_topic_threshold=SETTINGS.off_topic_threshold,
            groundedness_threshold=SETTINGS.groundedness_threshold,
            top_k=SETTINGS.answer_top_k,
        )
        if result.outcome != RagOutcome.INSUFFICIENT_EVIDENCE:
            continue
        insufficient_count += 1

        query_vector = embedder.embed_query(query_text)
        candidates = retriever.search(query_vector, k=SETTINGS.answer_top_k).scored_chunks
        if not candidates:
            continue

        synth_start = time.perf_counter()
        verdict = synthesize_answer_llm(query_text, [c.chunk.text for c in candidates])
        synth_ms = (time.perf_counter() - synth_start) * 1000.0
        synthesis_latencies_ms.append(synth_ms)

        f1 = word_overlap_f1(verdict.answer_text, gold_answer) if verdict.answerable else 0.0
        rows.append(
            {
                "query_id": int(row["query_id"]),
                "query": query_text,
                "gold_answer": gold_answer,
                "recovered": verdict.answerable,
                "synthesized_answer": verdict.answer_text,
                "f1": f1,
            }
        )
        print(f"  [{i + 1}/{sample_size}] processed", end="\r")

    print()
    report_df = pd.DataFrame(rows)
    n = len(report_df)
    recovered_n = int(report_df["recovered"].sum()) if n else 0

    lines = [
        f"# Multi-passage synthesis recovery eval — {collection_name}, "
        f"{sample_size} sampled queries, {insufficient_count} were INSUFFICIENT_EVIDENCE\n"
    ]
    if n == 0:
        lines.append("\nNo INSUFFICIENT_EVIDENCE rows in this sample — nothing to recover.\n")
    else:
        lines.append("\n## Recovery\n")
        lines.append("| metric | value |\n|---|---|\n")
        lines.append(f"| INSUFFICIENT_EVIDENCE rows attempted | {n} |\n")
        lines.append(f"| recovered (synthesis found an answer) | {recovered_n} ({recovered_n / n * 100:.1f}%) |\n")
        lines.append(
            f"| mean word-overlap F1 vs gold answer (recovered rows) | "
            f"{report_df[report_df['recovered']]['f1'].mean():.3f} |\n"
            if recovered_n
            else "| mean word-overlap F1 vs gold answer (recovered rows) | n/a |\n"
        )

        latency_stats = summarize(synthesis_latencies_ms)
        lines.append("\n## Synthesis call latency (async, NOT counted against the 200ms RAG SLA)\n")
        lines.append("| metric | value |\n|---|---|\n")
        lines.append(f"| P50 (ms) | {latency_stats['p50']:.1f} |\n")
        lines.append(f"| P70 (ms) | {latency_stats['p70']:.1f} |\n")
        lines.append(f"| P100 (ms) | {latency_stats['p100']:.1f} |\n")

    report_text = "".join(lines)
    output_path = SETTINGS.results_dir / f"synthesis_eval_{collection_name}.md"
    output_path.write_text(report_text)
    print(report_text)
    print(f"Saved to {output_path}")


def main() -> None:
    language = sys.argv[1] if len(sys.argv) > 1 else SETTINGS.default_language
    strategy = sys.argv[2] if len(sys.argv) > 2 else SETTINGS.default_strategy
    run_eval(language, strategy)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against an existing collection to verify**

Run: `uv run python -m scripts.run_synthesis_eval en native`
Expected: completes without error; prints and saves `results/synthesis_eval_en_native.md` showing how many of the sampled `INSUFFICIENT_EVIDENCE` rows synthesis recovers into real answers (and the F1 of those recovered answers vs. gold).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_synthesis_eval.py results/synthesis_eval_en_native.md
git commit -m "feat(eval): add multi-passage synthesis recovery eval for INSUFFICIENT_EVIDENCE queries"
```

---

### Task 10: Full regression pass

**Files:** none (verification only).

- [ ] **Step 1: Run the full fast test suite**

Run: `uv run pytest -m "not slow" -v`
Expected: all tests PASS (this includes every test touched or added in Tasks 1–6, plus the untouched Phase 1–4 suites, confirming nothing broke).

- [ ] **Step 2: Run the full suite including slow/real-API tests, if `ANTHROPIC_API_KEY` is configured**

Run: `uv run pytest -v`
Expected: all PASS, or slow tests SKIP with an `AnthropicError`/`anthropic API not configured` message if no key is set — never a hard failure.

- [ ] **Step 3: Re-run the three end-to-end eval scripts touched/added by this plan for all three languages, to refresh `results/`**

Run:
```bash
for lang in en hi ta; do
  uv run python -m scripts.run_answer_quality_eval "$lang" native
  uv run python -m scripts.run_synthesis_eval "$lang" native
done
```
Expected: all six invocations complete; `results/answer_quality_*.md` now include the "Real answer status among ANSWERED rows" table, and `results/synthesis_eval_*.md` exist for all three languages.

- [ ] **Step 4: Commit the refreshed results**

```bash
git add results/
git commit -m "chore: refresh eval results with real-answer-status and synthesis-recovery reports for hi/en/ta"
```
