# Phase 3: STT Integration & Async Groundedness Check — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Sarvam AI speech-to-text leg and a non-blocking Claude Sonnet 5 groundedness re-check on top of the existing, unchanged Phase 2 hot-path RAG pipeline, plus a three-way (STT/RAG/reflect) latency report.

**Architecture:** Two new independent modules (`stt.py`, `reflect.py`) called from a new CLI (`scripts/voice_answer.py`) that chains them around the existing `run_pipeline`. A separate script (`scripts/run_full_latency_eval.py`) measures STT and reflect latency independently and reuses Phase 2's committed RAG-only numbers verbatim.

**Tech Stack:** `httpx` (Sarvam REST call), `anthropic` SDK (`client.messages.parse`, structured output), `pydantic`, existing Phase 1/2 stack (`qdrant-client`, `sentence-transformers`).

## Global Constraints

- No hot-path changes: `run_pipeline`, `RagOutcome`, `pipeline_schemas.py`, and the existing guardrails from Phase 2 (`guardrails.py`, `generation.py`) are untouched by every task in this plan. Anything the reflect leg needs (source passage text) is obtained by a **separate, redundant local re-embed + re-retrieve call in the calling script**, not by adding fields to `Answer`/`PipelineResult` or otherwise touching `pipeline.py`/`pipeline_schemas.py`. This is a deliberate, conservative reading of "untouched" — flag to the user if this trade-off (a second embed+retrieve call, ~ms, on the non-hot-path leg only) is unwanted.
- The async groundedness check never blocks the returned answer or changes `RagOutcome` — informational only, logged/printed after the answer is already returned.
- No audio capture UI. STT takes a file path as input. No `sounddevice`/`pyaudio`.
- No combined end-to-end latency number. STT / RAG / async-check are three **separate** distributions.
- RAG-only latency numbers are reused **verbatim** from `results/latency_rag.md` (already committed): P50=7.16ms, P70=7.45ms, P100=43.17ms. Do not re-measure.
- `SARVAM_API_KEY` is read via `os.environ` at call time inside `stt.py` only — never stored on `Settings` (plain `pydantic.BaseModel`, holds no secrets).
- `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` / an `ant auth login` profile is resolved automatically by the zero-arg `anthropic.Anthropic()` client — never hardcode a key, never read it manually.
- Model ID for the groundedness check: `"claude-sonnet-5"`.
- Sarvam STT bounded retry: 2 attempts total, brief backoff, retry only on HTTP 429 or 5xx. Non-retryable status codes fail immediately via `response.raise_for_status()`.
- Anthropic call relies on the SDK's own default retry (`max_retries=2`) — no custom retry code around it.
- Missing `SARVAM_API_KEY` fails fast with a clear error identifying the missing env var (`SarvamAPIKeyMissingError`), raised before any network call.
- Bad/missing audio file path fails fast in `voice_answer.py` before any network call (`Path.is_file()` check).
- Config additions (all non-secret) go on the existing `Settings` model in `src/speculative_rag/config.py` — already done in this branch (see "Already Done" below). Every task in this plan can assume they exist.
- Sample sizes: `stt_latency_sample_size = 30`, `reflect_latency_sample_size = 30` (real API calls cost money — far smaller than the RAG-only leg's 200).
- BCP-47 language codes for the `{hi, en, ta}` set `LANGUAGE_SOURCES` uses: `hi -> hi-IN`, `en -> en-IN`, `ta -> ta-IN`. Already added as `BCP47_LANGUAGE_CODES` in `config.py`.
- Test convention (matches `tests/test_retrieval.py`): real-network tests are marked `@pytest.mark.slow` and must `pytest.skip(...)` gracefully (not fail) when credentials aren't configured — catch `SarvamAPIKeyMissingError` / `anthropic.AnthropicError` at the call site inside the test, not via `skipif` at collection time, since the anthropic SDK's own credential resolution can only be evaluated at call time.
- `voice_answer.py` and `run_full_latency_eval.py` are thin CLI/glue scripts, **untested directly** — matches the existing convention for `scripts/answer.py` and `scripts/run_latency_eval.py`.
- Verified against the installed SDK (`anthropic==0.122.0`) in this environment: `client.messages.parse(..., output_format=SomeModel)` returns a `ParsedMessage`; the parsed instance is at `response.parsed_output` (not `.parsed`). Use these exact names.

## Already Done (this session, on branch `phase3-stt-reflect`, commit `9b7faf8`)

- `pyproject.toml` / `uv.lock`: `anthropic>=0.122.0` and `httpx>=0.28.1` promoted to direct dependencies (`uv add anthropic httpx` already run — do not re-run).
- `src/speculative_rag/config.py`: added `sarvam_stt_url`, `sarvam_default_language_code`, `stt_latency_sample_size`, `reflect_latency_sample_size` to `Settings`, plus a new module-level `BCP47_LANGUAGE_CODES: dict[str, str]` constant.
- `tests/fixtures/sample_hindi.wav`: committed. A ~7s synthesized Hindi WAV (macOS `say -v Lekha`, 16-bit mono, 22050Hz), containing the phrase "नमस्ते, आप कैसे हैं? यह एक परीक्षण वाक्य है।" ("Hello, how are you? This is a test sentence."). Reused by both the `stt.py` integration test and the STT-only latency sample — do not regenerate it.
- `.gitignore`: added `.env` (an empty `.env` placeholder already existed untracked in the repo root, for the user to fill in `SARVAM_API_KEY` / `ANTHROPIC_API_KEY` later; it is not loaded automatically — no `python-dotenv` dependency exists or should be added, per the Global Constraint that `stt.py` reads `os.environ` directly).

## Task 1: `src/speculative_rag/stt.py` — Sarvam STT client

**Files:**
- Create: `src/speculative_rag/stt.py`
- Test: `tests/test_stt.py`

**Interfaces:**
- Consumes: `SETTINGS.sarvam_stt_url` (`str`, already in `config.py`).
- Produces: `transcribe_audio(file_path: Path, language_code: str) -> str` and `SarvamAPIKeyMissingError(RuntimeError)`, both imported by Task 3 and Task 4 as `from speculative_rag.stt import transcribe_audio, SarvamAPIKeyMissingError`.

Verified Sarvam REST shape (fetched from `https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe` this session):
- `POST https://api.sarvam.ai/speech-to-text`, `multipart/form-data`.
- Header: `api-subscription-key: <SARVAM_API_KEY>`.
- Form fields: `file` (the audio file), `language_code` (e.g. `"hi-IN"`).
- Response JSON (200): `{"transcript": "...", "language_code": "...", ...}` — only `transcript` is used.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stt.py
import os
from pathlib import Path

import httpx
import pytest

from speculative_rag.stt import SarvamAPIKeyMissingError, transcribe_audio

FIXTURE = Path(__file__).parent / "fixtures" / "sample_hindi.wav"


def _response(status_code: int, json_body: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.sarvam.ai/speech-to-text")
    return httpx.Response(status_code=status_code, json=json_body, request=request)


def test_transcribe_audio_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(SarvamAPIKeyMissingError):
        transcribe_audio(FIXTURE, "hi-IN")


def test_transcribe_audio_success_first_attempt(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    calls = []

    def fake_post(url, *, headers, data, files, timeout):
        calls.append((url, headers, data))
        return _response(200, {"transcript": "नमस्ते"})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = transcribe_audio(FIXTURE, "hi-IN")

    assert result == "नमस्ते"
    assert len(calls) == 1
    url, headers, data = calls[0]
    assert headers["api-subscription-key"] == "test-key"
    assert data["language_code"] == "hi-IN"


def test_transcribe_audio_retries_on_503_then_succeeds(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.setattr("speculative_rag.stt.time.sleep", lambda _seconds: None)
    responses = [_response(503), _response(200, {"transcript": "ok"})]

    def fake_post(url, *, headers, data, files, timeout):
        return responses.pop(0)

    monkeypatch.setattr(httpx, "post", fake_post)
    result = transcribe_audio(FIXTURE, "hi-IN")

    assert result == "ok"
    assert responses == []


def test_transcribe_audio_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.setattr("speculative_rag.stt.time.sleep", lambda _seconds: None)
    call_count = {"n": 0}

    def fake_post(url, *, headers, data, files, timeout):
        call_count["n"] += 1
        return _response(503)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError):
        transcribe_audio(FIXTURE, "hi-IN")
    assert call_count["n"] == 2


def test_transcribe_audio_raises_immediately_on_400_no_retry(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    call_count = {"n": 0}

    def fake_post(url, *, headers, data, files, timeout):
        call_count["n"] += 1
        return _response(400, {"error": "bad request"})

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError):
        transcribe_audio(FIXTURE, "hi-IN")
    assert call_count["n"] == 1


@pytest.mark.slow
def test_transcribe_audio_real_call():
    try:
        transcript = transcribe_audio(FIXTURE, "hi-IN")
    except SarvamAPIKeyMissingError:
        pytest.skip("SARVAM_API_KEY not set")
    assert isinstance(transcript, str)
    assert transcript.strip()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stt.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'speculative_rag.stt'`

- [ ] **Step 3: Write the implementation**

```python
# src/speculative_rag/stt.py
"""Sarvam AI speech-to-text integration (Phase 3 STT leg). Wraps the Sarvam
speech-to-text REST endpoint as a bounded-retry multipart POST. Takes an
audio file path as input only — no live mic handling here (spec non-goal)."""

import os
import time
from pathlib import Path

import httpx

from speculative_rag.config import SETTINGS

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 2
_BACKOFF_SECONDS = 1.0


class SarvamAPIKeyMissingError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "SARVAM_API_KEY is not set in the environment. "
            "Set it before calling transcribe_audio()."
        )


def transcribe_audio(file_path: Path, language_code: str) -> str:
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise SarvamAPIKeyMissingError()

    headers = {"api-subscription-key": api_key}
    data = {"language_code": language_code}

    last_error: httpx.HTTPStatusError | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        with file_path.open("rb") as audio_file:
            files = {"file": (file_path.name, audio_file, "audio/wav")}
            response = httpx.post(
                SETTINGS.sarvam_stt_url,
                headers=headers,
                data=data,
                files=files,
                timeout=30.0,
            )

        if response.status_code not in _RETRYABLE_STATUS_CODES:
            response.raise_for_status()
            return response.json()["transcript"]

        last_error = httpx.HTTPStatusError(
            f"Sarvam STT returned {response.status_code} after {attempt} attempt(s)",
            request=response.request,
            response=response,
        )
        if attempt < _MAX_ATTEMPTS:
            time.sleep(_BACKOFF_SECONDS)

    assert last_error is not None
    raise last_error
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stt.py -v -m "not slow"`
Expected: 5 passed (the `slow` test is deselected; it needs `SARVAM_API_KEY` and is exercised manually once the key is available)

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/stt.py tests/test_stt.py
git commit -m "feat(phase3): add Sarvam STT client with bounded retry"
```

---

## Task 2: `src/speculative_rag/reflect.py` — async groundedness check

**Files:**
- Create: `src/speculative_rag/reflect.py`
- Test: `tests/test_reflect.py`

**Interfaces:**
- Produces: `ReflectVerdict(pydantic.BaseModel)` with fields `grounded: bool`, `reasoning: str`; and `check_groundedness_llm(answer_text: str, source_text: str) -> ReflectVerdict`. Both imported by Task 3 and Task 4 as `from speculative_rag.reflect import ReflectVerdict, check_groundedness_llm`.
- This module is purely additive: never imported by `pipeline.py`, never consulted inside `run_pipeline`.

Verified against the installed `anthropic==0.122.0` SDK in this environment (via `inspect.signature` and reading `anthropic/types/parsed_message.py` directly): `client.messages.parse(model=..., max_tokens=..., messages=..., output_format=SomeModel)` returns a `ParsedMessage`; the parsed model instance is at `.parsed_output` (a property that scans the response content for a text block's `parsed_output`). Use `output_format=`, not `output_config=`. Use `.parsed_output`, not `.parsed`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reflect.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reflect.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'speculative_rag.reflect'`

- [ ] **Step 3: Write the implementation**

```python
# src/speculative_rag/reflect.py
"""Async LLM-based groundedness check (spec: rho_Self-reflect). Purely
additive, non-hot-path: never imported by pipeline.py, never consulted
inside run_pipeline, never changes RagOutcome. A secondary, informational
verdict on whether an already-returned ANSWERED result is actually
grounded in its source passage, using Claude Sonnet 5 structured output."""

import anthropic
from pydantic import BaseModel

_MODEL = "claude-sonnet-5"

_PROMPT_TEMPLATE = """You are reviewing whether an answer is grounded in its source passage.

Source passage:
{source_text}

Answer:
{answer_text}

Decide whether the answer is fully supported by the source passage (grounded=true) \
or introduces claims not present in the source passage (grounded=false). Give a \
brief one or two sentence reasoning for your verdict."""


class ReflectVerdict(BaseModel):
    grounded: bool
    reasoning: str


def check_groundedness_llm(answer_text: str, source_text: str) -> ReflectVerdict:
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": _PROMPT_TEMPLATE.format(
                    source_text=source_text, answer_text=answer_text
                ),
            }
        ],
        output_format=ReflectVerdict,
    )
    verdict = response.parsed_output
    assert verdict is not None
    return verdict
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reflect.py -v -m "not slow"`
Expected: 2 passed (the `slow` test is deselected; it needs `ANTHROPIC_API_KEY`/an `ant auth login` profile and is exercised manually once configured)

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/reflect.py tests/test_reflect.py
git commit -m "feat(phase3): add async Claude Sonnet 5 groundedness check"
```

---

## Task 3: `scripts/voice_answer.py` — voice-in CLI

**Depends on:** Task 1 (`stt.transcribe_audio`), Task 2 (`reflect.check_groundedness_llm`). Do not start until both are merged.

**Files:**
- Create: `scripts/voice_answer.py`
- No test file — thin CLI/glue script, untested directly (Global Constraints; matches `scripts/answer.py` convention).

**Interfaces:**
- Consumes: `transcribe_audio(file_path: Path, language_code: str) -> str` (Task 1); `check_groundedness_llm(answer_text: str, source_text: str) -> ReflectVerdict` with `.grounded: bool`, `.reasoning: str` (Task 2); `BCP47_LANGUAGE_CODES: dict[str, str]` and `SETTINGS` (`config.py`, already present); `run_pipeline(...)`, `RagOutcome` (`pipeline.py`/`pipeline_schemas.py`, unchanged, same signature `scripts/answer.py` already uses).

- [ ] **Step 1: Write the implementation**

```python
# scripts/voice_answer.py
"""Voice-in extractive-answer CLI (Phase 3): an audio file path is
transcribed via Sarvam STT, then fed unchanged into the Phase 2 hot-path
pipeline, then (only if ANSWERED) reviewed by a non-blocking, informational
Claude Sonnet 5 groundedness check. The reflect leg re-embeds and
re-retrieves the top chunk locally to get the source passage text, rather
than touching pipeline.py/pipeline_schemas.py (see plan Global Constraints).

Usage: uv run python -m scripts.voice_answer <audio_file> [language] [strategy]
"""

import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient

from speculative_rag.config import BCP47_LANGUAGE_CODES, SETTINGS
from speculative_rag.embedding import Embedder
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.reflect import check_groundedness_llm
from speculative_rag.retrieval import Retriever
from speculative_rag.stt import transcribe_audio


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python -m scripts.voice_answer <audio_file> [language] [strategy]")
        return

    audio_path = Path(sys.argv[1])
    if not audio_path.is_file():
        print(f"No such audio file: {audio_path}")
        return

    language = sys.argv[2] if len(sys.argv) > 2 else SETTINGS.default_language
    strategy = sys.argv[3] if len(sys.argv) > 3 else SETTINGS.default_strategy
    collection_name = f"{language}_{strategy}"
    language_code = BCP47_LANGUAGE_CODES.get(language, SETTINGS.sarvam_default_language_code)

    print(f"Transcribing '{audio_path}' (language_code={language_code})...")
    stt_start = time.perf_counter()
    transcript = transcribe_audio(audio_path, language_code)
    stt_ms = (time.perf_counter() - stt_start) * 1000.0
    print(f"transcript: {transcript}")
    print(f"stt latency (ms): {stt_ms:.2f}\n")

    print(f"Loading embedder and connecting to collection '{collection_name}'...")
    embedder = Embedder()
    client = QdrantClient(path=str(SETTINGS.qdrant_path))
    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        print(f"No such collection '{collection_name}'. Available: {available}")
        print("Run `uv run python -m scripts.run_recall_eval` first to build the index.")
        return
    retriever = Retriever(client, collection_name)

    result = run_pipeline(
        transcript,
        embedder,
        retriever,
        off_topic_threshold=SETTINGS.off_topic_threshold,
        groundedness_threshold=SETTINGS.groundedness_threshold,
        top_k=SETTINGS.answer_top_k,
    )

    print(f"outcome: {result.outcome.value}")
    if result.outcome == RagOutcome.ANSWERED:
        print(f"answer: {result.answer.text}")
        print(
            f"source passage: {result.answer.source_passage_id}  "
            f"score={result.answer.retrieval_score:.3f}  "
            f"groundedness={result.answer.groundedness_score:.3f}"
        )
    else:
        print(f"refused: {result.refusal_reason}")

    print("latency (ms):")
    for timing in result.timings:
        print(f"  {timing.stage:24s} {timing.elapsed_ms:7.2f}")
    print(f"  {'TOTAL':24s} {result.total_ms:7.2f}\n")

    if result.outcome == RagOutcome.ANSWERED:
        print("--- review (informational, non-blocking; answer above is already final) ---")
        query_vector = embedder.embed_query(transcript)
        retrieval = retriever.search(query_vector, k=1)
        source_text = retrieval.top.chunk.text if retrieval.top is not None else ""

        reflect_start = time.perf_counter()
        verdict = check_groundedness_llm(result.answer.text, source_text)
        reflect_ms = (time.perf_counter() - reflect_start) * 1000.0
        print(f"grounded: {verdict.grounded}")
        print(f"reasoning: {verdict.reasoning}")
        print(f"async groundedness check latency (ms): {reflect_ms:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual smoke test (no automated test — thin glue script)**

Run: `uv run python -m scripts.voice_answer --help` first to confirm it imports cleanly:
`uv run python -c "import scripts.voice_answer"`
Expected: no import errors (fails fast on missing collection/audio only at runtime, not at import time).

If `SARVAM_API_KEY`/`ANTHROPIC_API_KEY` and an existing `hi_native` collection are available, run:
`uv run python -m scripts.voice_answer tests/fixtures/sample_hindi.wav hi native`
Expected: prints transcript, pipeline outcome, and (if ANSWERED) a review section — verify manually since this is a thin CLI script per project convention.

- [ ] **Step 3: Commit**

```bash
git add scripts/voice_answer.py
git commit -m "feat(phase3): add voice_answer CLI chaining STT -> pipeline -> async reflect"
```

---

## Task 4: `scripts/run_full_latency_eval.py` — three-way latency report

**Depends on:** Task 1 (`stt.transcribe_audio`), Task 2 (`reflect.check_groundedness_llm`). Do not start until both are merged. Independent of Task 3 (disjoint files) — may run in parallel with it once Tasks 1-2 are merged.

**Files:**
- Create: `scripts/run_full_latency_eval.py`
- No test file — thin CLI/glue script, untested directly (Global Constraints; matches `scripts/run_latency_eval.py` convention).

**Interfaces:**
- Consumes: `transcribe_audio` (Task 1); `check_groundedness_llm` (Task 2); `summarize(values: list[float]) -> dict[str, float]` with keys `p50`/`p70`/`p100` (`latency.py`, unchanged — same function `scripts/run_latency_eval.py` already uses); `load_validation_sample`, `build_query_records` (`data.py`, unchanged, same signatures `scripts/run_latency_eval.py` already uses); `run_pipeline`, `RagOutcome` (unchanged).
- Writes `results/latency_full.md` (new file; does not touch `results/latency_rag.md`).

- [ ] **Step 1: Write the implementation**

```python
# scripts/run_full_latency_eval.py
"""Three-way latency report (spec Sec 7, items 2-3): STT-only, RAG-only
(reused verbatim from Phase 2's results/latency_rag.md, NOT re-measured),
and async-groundedness-only. Each leg is an independent distribution --
no combined end-to-end number (spec explicitly rules this out). Costs real
money per request (Sarvam + Anthropic calls), so sample sizes are
SETTINGS.stt_latency_sample_size / reflect_latency_sample_size (30 each),
far smaller than the RAG-only leg's 200.

Usage: uv run python -m scripts.run_full_latency_eval
"""

import time
from pathlib import Path

from qdrant_client import QdrantClient

from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS
from speculative_rag.data import build_query_records, load_validation_sample
from speculative_rag.embedding import Embedder
from speculative_rag.latency import summarize
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.reflect import check_groundedness_llm
from speculative_rag.retrieval import Retriever
from speculative_rag.stt import transcribe_audio

# Reused verbatim from Phase 2's results/latency_rag.md (200 validation
# queries, hi_native collection) -- NOT re-measured here.
RAG_ONLY_P50_MS = 7.16
RAG_ONLY_P70_MS = 7.45
RAG_ONLY_P100_MS = 43.17
RAG_ONLY_SAMPLE_SIZE = 200

STT_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_hindi.wav"
STT_FIXTURE_LANGUAGE_CODE = "hi-IN"


def _language_source(language: str):
    return next(source for source in LANGUAGE_SOURCES if source.language == language)


def measure_stt_only(sample_size: int) -> list[float]:
    latencies = []
    for _ in range(sample_size):
        start = time.perf_counter()
        transcribe_audio(STT_FIXTURE_PATH, STT_FIXTURE_LANGUAGE_CODE)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def measure_reflect_only(sample_size: int, embedder, retriever) -> list[float]:
    source = _language_source(SETTINGS.default_language)
    # Over-sample: not every validation query reaches ANSWERED, and only
    # ANSWERED results have a groundedness check to time.
    df = load_validation_sample(source.parquet_path, sample_size * 2, SETTINGS.sample_seed)
    records = build_query_records(df, source.language, source.text_field)

    latencies: list[float] = []
    for record in records:
        if len(latencies) >= sample_size:
            break
        result = run_pipeline(
            record.query_text,
            embedder,
            retriever,
            off_topic_threshold=SETTINGS.off_topic_threshold,
            groundedness_threshold=SETTINGS.groundedness_threshold,
            top_k=SETTINGS.answer_top_k,
        )
        if result.outcome != RagOutcome.ANSWERED:
            continue

        query_vector = embedder.embed_query(record.query_text)
        retrieval = retriever.search(query_vector, k=1)
        source_text = retrieval.top.chunk.text if retrieval.top is not None else ""

        start = time.perf_counter()
        check_groundedness_llm(result.answer.text, source_text)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def render_report(stt_latencies: list[float], reflect_latencies: list[float]) -> str:
    stt_stats = summarize(stt_latencies)
    reflect_stats = summarize(reflect_latencies)
    lines = [
        "# Three-way latency report -- STT / RAG / async-groundedness (spec Sec 7, items 2-3)",
        "",
        "Each leg below is an independent distribution; per spec, there is no combined",
        "end-to-end latency number.",
        "",
        f"## STT-only ({len(stt_latencies)} calls, Sarvam speech-to-text)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| P50 (ms) | {stt_stats['p50']:.2f} |",
        f"| P70 (ms) | {stt_stats['p70']:.2f} |",
        f"| P100 (ms) | {stt_stats['p100']:.2f} |",
        "",
        f"## RAG-only (reused verbatim from Phase 2's results/latency_rag.md, {RAG_ONLY_SAMPLE_SIZE} validation queries -- not re-measured)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| P50 (ms) | {RAG_ONLY_P50_MS:.2f} |",
        f"| P70 (ms) | {RAG_ONLY_P70_MS:.2f} |",
        f"| P100 (ms) | {RAG_ONLY_P100_MS:.2f} |",
        "",
        f"## Async-groundedness-only ({len(reflect_latencies)} calls, Claude Sonnet 5, non-blocking)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| P50 (ms) | {reflect_stats['p50']:.2f} |",
        f"| P70 (ms) | {reflect_stats['p70']:.2f} |",
        f"| P100 (ms) | {reflect_stats['p100']:.2f} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print(f"Measuring STT-only latency ({SETTINGS.stt_latency_sample_size} calls against the committed fixture)...")
    stt_latencies = measure_stt_only(SETTINGS.stt_latency_sample_size)

    language = SETTINGS.default_language
    strategy = SETTINGS.default_strategy
    collection_name = f"{language}_{strategy}"
    embedder = Embedder()
    client = QdrantClient(path=str(SETTINGS.qdrant_path))
    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        print(f"No such collection '{collection_name}'. Available: {available}")
        print("Run `uv run python -m scripts.run_recall_eval` first to build the index.")
        return
    retriever = Retriever(client, collection_name)

    print(f"Measuring async-groundedness-only latency ({SETTINGS.reflect_latency_sample_size} calls)...")
    reflect_latencies = measure_reflect_only(SETTINGS.reflect_latency_sample_size, embedder, retriever)

    report = render_report(stt_latencies, reflect_latencies)
    SETTINGS.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = SETTINGS.results_dir / "latency_full.md"
    output_path.write_text(report)
    print(report)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual smoke test (no automated test — thin glue script)**

Run: `uv run python -c "import scripts.run_full_latency_eval"`
Expected: no import errors.

If `SARVAM_API_KEY`/`ANTHROPIC_API_KEY` and an existing `hi_native` collection are available, run:
`uv run python -m scripts.run_full_latency_eval`
Expected: prints and writes `results/latency_full.md` with three populated sections. This is the deferred spec §7 items 2-3 report — run it once credentials are available and commit the resulting `results/latency_full.md`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_full_latency_eval.py
git commit -m "feat(phase3): add three-way STT/RAG/reflect latency report script"
```

---

## Self-Review Notes (completed by the plan author)

- **Spec coverage:** STT integration (Task 1, 3) — covered. Async groundedness check (Task 2, 3) — covered. Three-way latency report (Task 4) — covered, RAG-only reused verbatim per spec. Config additions — already done. New deps (`anthropic`, `httpx` direct) — already done. Error handling (missing key, bad audio path, bounded retry, SDK default retry) — covered in Tasks 1 and 3. Test fixture resolution order — resolved: `tests/fixtures/` was empty, so step 2 of the spec's resolution order applied (macOS `say -v Lekha`, Hindi-capable); already generated and committed.
- **Placeholder scan:** no TBD/TODO; all steps carry complete code.
- **Type consistency:** `transcribe_audio(file_path: Path, language_code: str) -> str` matches its Task 1 definition and Task 3/4 call sites. `check_groundedness_llm(answer_text: str, source_text: str) -> ReflectVerdict` matches across Task 2 definition and Task 3/4 call sites. `ReflectVerdict.grounded`/`.reasoning` used consistently.
- **Deviation flagged for the user:** the design doc's architecture diagram doesn't mention re-embedding/re-retrieving to get source passage text for the reflect call — the design left `Answer`'s exact fields as of Phase 2. Since `Answer` has no source-passage-text field and the design explicitly says `pipeline.py`/`run_pipeline` are untouched, Tasks 3 and 4 re-run `embedder.embed_query` + `retriever.search(k=1)` after an `ANSWERED` outcome to fetch the chunk text, rather than extending `Answer`. This is a few extra milliseconds of local, non-network computation on the non-hot-path leg only.
