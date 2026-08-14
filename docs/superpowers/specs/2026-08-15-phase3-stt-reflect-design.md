# Phase 3: Speech-to-Text Integration & Async Groundedness Check — Design

**Status:** Approved by user, 2026-08-15. Ready for implementation planning.

## Goal

Close out the two items Phase 2 explicitly deferred (see `docs/superpowers/plans/2026-08-15-phase2-hotpath-guardrails-generation.md` Global Constraints):

1. **STT integration** — transcribe spoken queries via Sarvam AI's speech-to-text REST API, feeding the transcript into the existing Phase 2 hot-path RAG pipeline unchanged.
2. **The async LLM-based groundedness check (`ρ_Self-reflect`)** — a non-hot-path, secondary semantic verdict on whether an `ANSWERED` result is actually grounded in its source passage, using Claude Sonnet 5.
3. **The full spec §7 three-way latency report** — STT-only, RAG-only (already delivered in Phase 2), and async-groundedness-only P50/P70/P100, each measured and reported independently.

## Non-goals (explicitly out of scope this phase)

- **No audio capture UI.** The user is building a separate JS app for microphone recording/upload. This phase's STT integration takes an audio **file path** as input — trivially callable by whatever hands off the recording later (that JS app, or a future thin HTTP wrapper). No `sounddevice`/`pyaudio`, no live mic handling.
- **No hot-path changes.** `run_pipeline`, `RagOutcome`, and the existing guardrails from Phase 2 are untouched. The async groundedness check never blocks the returned answer or changes `RagOutcome` — it is reported as an independent, informational secondary verdict, logged after the fact.
- **No combined end-to-end latency number.** Per spec §7, STT / RAG / async-check are reported as three **separate** distributions, not one fused pipeline latency. RAG-only numbers are reused from Phase 2's `results/latency_rag.md`, not re-measured.
- **No structured-logging infrastructure** beyond what's needed to compute and print the latency numbers (same scoping Phase 2 used).
- **No retries/backoff framework** — just a small bounded retry (2 attempts) hand-rolled into the Sarvam client, since that's now a real network leg; the Anthropic SDK's built-in retry covers the LLM call for free.

## Architecture

Two new independent legs bolt onto the existing Phase 2 pipeline, chained by one new CLI:

```
scripts/voice_answer.py:
  audio file path → stt.transcribe_audio()        [Sarvam STT, timed]
                  → pipeline.run_pipeline()         [existing Phase 2 hot path, timed, UNCHANGED]
                  → (if ANSWERED) reflect.check_groundedness_llm()
                                                     [Claude Sonnet 5, timed, non-blocking —
                                                      the answer has already been returned]
```

### New modules

- **`src/speculative_rag/stt.py`** — `transcribe_audio(file_path: Path, language_code: str) -> str`.
  Wraps Sarvam's `POST https://api.sarvam.ai/speech-to-text` REST endpoint via `httpx` (currently a transitive dependency via `qdrant-client`; promoted to a direct dependency this phase) as a multipart form POST with the audio file and `language_code`. Reads `SARVAM_API_KEY` from the environment (`os.environ`) at call time — never stored in `Settings`, which is a plain `pydantic.BaseModel` (not `BaseSettings`) and holds no secrets today. Raises a clear error if the env var is unset, rather than silently failing. Includes a small bounded retry (2 attempts, brief backoff) on 429/5xx responses.

- **`src/speculative_rag/reflect.py`** — `check_groundedness_llm(answer_text: str, source_text: str) -> ReflectVerdict`.
  `ReflectVerdict` is a new pydantic model: `grounded: bool`, `reasoning: str`. Uses the `anthropic` SDK's zero-arg client (`anthropic.Anthropic()`, resolves `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`/an `ant auth login` profile automatically — never hardcode a key). Calls `model="claude-sonnet-5"` with `output_config.format` (structured JSON output) so `grounded`/`reasoning` return typed via `client.messages.parse()`, not parsed from free text. This is purely additive — it is never consulted by `run_pipeline` and never changes `RagOutcome`.

- **`scripts/voice_answer.py`** — the CLI, parallel in shape to `scripts/answer.py`:
  `uv run python -m scripts.voice_answer <audio_file> [language] [strategy]`
  Prints the STT transcript + its latency, then the existing Phase 2 answer output (unchanged format), then — only if the outcome was `ANSWERED` — a separate "review" section with the async groundedness verdict + its own latency, clearly labeled as informational/non-blocking.

### Config additions (`src/speculative_rag/config.py`)

Added to the existing `Settings` model (all non-secret):
- `sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"`
- `sarvam_default_language_code: str = "hi-IN"` (BCP-47; the plan should derive the right code from `default_language` when `voice_answer.py` is invoked with a different `[language]` arg — implementer maps the small `{hi, en, ta}` set already used by `LANGUAGE_SOURCES` to their BCP-47 equivalents)
- `stt_latency_sample_size: int = 30`
- `reflect_latency_sample_size: int = 30`

(Sample sizes are far smaller than `latency_sample_size = 200` because Sarvam and Anthropic calls cost real money per request, unlike Phase 2's fully local RAG measurement.)

## Latency measurement — the deferred three-way report

Spec §7 items 2-3 want three **separate** distributions:

1. **STT-only**: implementer sources or records one short (~5-10s) Hindi test WAV, calls `transcribe_audio` on it `stt_latency_sample_size` (30) times, reports P50/P70/P100 of the network+transcription round trip.
2. **RAG-only**: already delivered in Phase 2 (`results/latency_rag.md`: P50=7.16ms, P70=7.45ms, P100=43.17ms). Reused verbatim in the Phase 3 report — not re-measured, since it's an independent leg unaffected by STT/reflect now existing.
3. **Async-groundedness-only**: run `check_groundedness_llm` on `reflect_latency_sample_size` (30) `ANSWERED` results (a fresh validation-set sample, or reuse a subset from Phase 2's run), report P50/P70/P100 of the Claude Sonnet 5 round trip.

All three land in a new `results/latency_full.md`, produced by `scripts/run_full_latency_eval.py` — a separate script from Phase 2's `scripts/run_latency_eval.py` (which remains untouched and independently reproducible without the new Sarvam/Anthropic dependencies).

## Error handling

- **Sarvam STT**: bounded retry (2 attempts, brief backoff) on 429/5xx inside `stt.py`. A real network leg now exists, unlike Phase 2's reasoning for deferring retries.
- **Anthropic call**: SDK's built-in retry (`max_retries=2` default) — no custom code needed.
- **Missing `SARVAM_API_KEY`**: fail fast in `stt.py` with a clear error message identifying the missing env var; `voice_answer.py` does not silently skip the STT leg.
- **Bad/missing audio file path**: fail fast in `voice_answer.py` before making any network call.

## Testing

- `stt.py` and `reflect.py` get `@pytest.mark.slow` integration tests (real API calls, matching Phase 2's `test_retrieval.py` convention) — plus, where feasible, a pure-logic unit test for request/response shaping that doesn't require a real network call (e.g. multipart body construction, `ReflectVerdict` parsing from a canned response).
- `voice_answer.py` and `run_full_latency_eval.py` are thin CLI/glue scripts, untested directly — same convention Phase 1/2 used for `scripts/query.py`, `scripts/answer.py`, `scripts/run_recall_eval.py`, `scripts/run_latency_eval.py` — validated instead by manual smoke-test steps in the implementation plan.

## Dependencies

New direct dependencies (via `uv add`): `anthropic`, and promoting the already-transitive `httpx` to direct. No other new runtime dependencies.

## Prerequisites

- `SARVAM_API_KEY` — user confirmed they already have this; will be added to the environment before the STT task's integration test runs.
- `ANTHROPIC_API_KEY` (or an `ant auth login` profile) — resolved automatically by the `anthropic` SDK's zero-arg client.
- **Test audio clip**: none on hand yet, and an implementer subagent has no microphone, so it cannot record one. Resolution order for the plan to specify explicitly:
  1. Check `tests/fixtures/` for a user-provided WAV first (the human may drop one in before the STT task runs).
  2. If absent, synthesize one via a local TTS tool available in the dev environment (e.g. macOS `say`, checked via `say -v '?'` for a Hindi-capable voice) and save it to `tests/fixtures/sample_hindi.wav`, committed to the repo as a fixture.
  3. If no local TTS with Hindi support is available either, fall back to a short English phrase via `say` (or any available TTS) — Sarvam's `transcribe` mode accepts `language_code: "unknown"` for auto-detect, so the integration test can still validate the real API round-trip (auth, multipart upload, response parsing) even if the fixture isn't Hindi. Note this limitation in the task's self-review rather than silently treating an English fixture as satisfying the Hindi-corpus intent.
  This applies to both the `stt.py` integration test and the STT-only latency sample in `run_full_latency_eval.py` — both reuse the same fixture file.
