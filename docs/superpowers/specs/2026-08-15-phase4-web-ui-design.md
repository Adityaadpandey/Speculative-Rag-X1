# Phase 4: Interactive Web UI — Design

**Status:** Approved by user, 2026-08-15. Ready for implementation planning.

## Goal

Give the RAG pipeline (Phase 2 hot path + Phase 3 STT/reflect legs) an interactive local web UI: type a question or provide audio (file upload or live mic recording), see the extractive answer with its guardrail outcome and per-stage latency, and — for `ANSWERED` results — a non-blocking async groundedness review that fills in after the answer is already shown.

This supersedes the Phase 3 design doc's stated non-goal ("no audio capture UI ... a separate JS app for microphone recording/upload") — the user now wants that capability built in-repo rather than as a separate app.

## Non-goals (explicitly out of scope this phase)

- No authentication/multi-user support. Single local user, single local server process.
- No persistence across page refreshes. "Latest result only" — no database, no query history, no session storage.
- No changes to `src/speculative_rag/*`. The web app is a pure HTTP layer over the existing, unchanged package (`pipeline.run_pipeline`, `stt.transcribe_audio`, `reflect.check_groundedness_llm`, `Embedder`, `Retriever`).
- No websockets/SSE. The "non-blocking" feel for the groundedness review comes from the frontend calling `/api/reflect` as a second, separate `fetch()` after `/api/query` already resolved and rendered — plain sequential async HTTP, no streaming protocol.
- No audio longer than Sarvam's 30s synchronous-endpoint limit. The mic recorder auto-stops at 30s; no chunking, no batch-API fallback.
- No frontend build step, no npm/node toolchain. Plain HTML/CSS/JS served as static files.
- No changes to the existing CLI scripts (`scripts/answer.py`, `scripts/voice_answer.py`, `scripts/run_latency_eval.py`, `scripts/run_full_latency_eval.py`) — all remain as-is, independently usable.

## Architecture

```
scripts/serve_ui.py          thin launcher: `uv run python -m scripts.serve_ui`
                              runs `uvicorn webapp.main:app`

webapp/main.py                FastAPI app. Loads Embedder + one Retriever per
                               available collection ONCE at startup (module-level
                               state), reused across requests — same "load once,
                               serve many queries" pattern scripts/answer.py uses
                               in its input loop.

webapp/static/
  index.html                  page structure
  app.js                      fetch() calls to the API, MediaRecorder for mic,
                               DOM updates for the two-phase (query, then reflect) flow
  style.css                   styling
```

### New dependencies (via `uv add`)

`fastapi`, `uvicorn[standard]`, `python-multipart` (required by FastAPI for multipart file/form uploads — used by `/api/transcribe`).

## API surface

All endpoints return JSON except `GET /` and `GET /static/*`.

### `GET /`
Serves `webapp/static/index.html`.

### `GET /api/collections`
No params. Returns:
```json
{
  "collections": [{"language": "hi", "strategy": "native"}, ...],
  "sarvam_key_configured": true,
  "anthropic_key_configured": false
}
```
`collections` is derived from `client.get_collections()` at startup (cached — the index is read-only and doesn't change during the server's lifetime, matching every existing script's read-only assumption), parsed from each collection's `{language}_{strategy}` name using `LANGUAGE_SOURCES` languages and the four known strategy names. The two `*_key_configured` booleans are `bool(os.environ.get("SARVAM_API_KEY"))` / `bool(os.environ.get("ANTHROPIC_API_KEY"))` — checked at request time (env can change if the server is restarted with a populated `.env`), never the values themselves. The frontend calls this once on page load to populate the language/strategy dropdowns and to decide whether to disable audio/mic controls.

### `POST /api/query`
Request body:
```json
{"query_text": "...", "language": "hi", "strategy": "native"}
```
Looks up the pre-loaded `Retriever` for `{language}_{strategy}`; 404 (`{"error": "..."}`) if that collection isn't loaded. Runs `run_pipeline` with `SETTINGS.off_topic_threshold`, `SETTINGS.groundedness_threshold`, `SETTINGS.answer_top_k` — identical call to `scripts/answer.py`. Response:
```json
{
  "outcome": "answered",
  "answer": {"text": "...", "source_passage_id": "...", "retrieval_score": 0.91, "groundedness_score": 0.6},
  "refusal_reason": null,
  "timings": [{"stage": "embed", "elapsed_ms": 1.2}, ...],
  "total_ms": 7.3
}
```
`answer` is `null` when `outcome != "answered"`; `refusal_reason` is `null` when it is.

### `POST /api/transcribe`
Multipart form: `file` (the audio blob/file), `language` (the `hi`/`en`/`ta` code — mapped server-side to BCP-47 via the existing `BCP47_LANGUAGE_CODES`, same as `voice_answer.py`). Calls `transcribe_audio`. Response:
```json
{"transcript": "...", "stt_latency_ms": 812.4}
```
On `SarvamAPIKeyMissingError`, returns HTTP 400 with `{"error": "SARVAM_API_KEY is not set..."}` (the frontend shouldn't normally hit this, since `/api/collections` already disabled the controls — but the check stays server-side too, since the key could be removed between page load and submission).

`transcribe_audio(file_path: Path, language_code: str)`'s signature is unchanged (Phase 3 constraint), but FastAPI hands the handler an `UploadFile` (in-memory/spooled bytes), not a `Path`. The handler writes the uploaded bytes to a `tempfile.NamedTemporaryFile(suffix=..., delete=False)` (suffix taken from the upload's content-type, e.g. `.webm` for mic recordings or the original filename's extension for uploads), calls `transcribe_audio` with that path, then deletes the temp file in a `finally` block — no change to `stt.py`.

### `POST /api/reflect`
Calls `check_groundedness_llm`. Response:
```json
{"grounded": true, "reasoning": "...", "reflect_latency_ms": 1450.2}
```
`Answer` doesn't carry chunk text (Phase 3 constraint — `pipeline.py`/`pipeline_schemas.py` stay untouched), so `source_text` is obtained the same way `voice_answer.py` does it: the **backend** re-embeds the original query text and re-retrieves top-1 from the same collection to get `chunk.text`, right before calling `check_groundedness_llm`. The actual request body therefore carries the original `query_text` and `language`/`strategy` (to know which collection to re-query) instead of `source_text` directly:
```json
{"query_text": "...", "language": "hi", "strategy": "native", "answer_text": "..."}
```
This keeps the re-embed/re-retrieve logic server-side (reusing the already-loaded `Embedder`/`Retriever`) instead of shipping chunk text to the browser and back, and mirrors `voice_answer.py`'s existing pattern exactly.

### Error handling (all endpoints)
Unhandled exceptions return HTTP 500 with `{"error": str(exc)}` via a FastAPI exception handler — the frontend always gets JSON back, never an HTML error page, so it can render an inline message instead of breaking.

## Frontend behavior

Single page, top to bottom:

1. **Language / strategy dropdowns** — populated from `/api/collections`, defaulting to `hi`/`native` (matches `SETTINGS.default_language`/`default_strategy`).
2. **Text input row** — text box + "Ask" button. Submits straight to `/api/query`.
3. **Audio input row** — file picker ("upload audio") and a record/stop toggle button ("record", using `MediaRecorder`, auto-stopping at 30s with a visible countdown). Both feed into the same flow: `POST /api/transcribe` (fills the text box with the resulting transcript, visibly, so the user sees what was heard) → then automatically `POST /api/query` with that transcript. If `sarvam_key_configured` is `false`, both controls are disabled with a tooltip ("SARVAM_API_KEY not set — text queries only").
4. **Result panel** — appears after `/api/query` resolves: outcome badge (`answered` / `off_topic_refused` / `unsafe_input_refused` / `ungrounded_refused`, color-coded), answer text + source passage id + retrieval/groundedness scores (if `answered`), or the refusal reason (if not), and a latency table listing every stage from `timings` plus `TOTAL`.
5. **Review panel** — only rendered when `outcome == "answered"`. Immediately shows a "checking groundedness..." spinner and fires `POST /api/reflect` in the background (not awaited before the result panel renders — the result panel is already final and interactive). When it resolves: grounded/not-grounded badge, reasoning text, its own latency. If `anthropic_key_configured` is `false`, this panel instead shows a static "ANTHROPIC_API_KEY not set — skipping groundedness review" message and never calls the endpoint. If the call fails for another reason, an inline error replaces the spinner — the result panel above is untouched.

Each new query resets the result and review panels — no history retained across queries or refreshes (Non-goals).

## Testing

- `tests/test_webapp.py` — FastAPI `TestClient`:
  - `/api/collections`: fake collection list + monkeypatched env vars → correct booleans and collection parsing.
  - `/api/query`: fake `Embedder`/`Retriever` injected (same fakes-over-real-model pattern `tests/test_pipeline.py` uses) → correct JSON shape for an `answered` and a `refused` case.
  - `/api/transcribe`: monkeypatch `webapp.main.transcribe_audio` → correct JSON shape; missing-key case → HTTP 400.
  - `/api/reflect`: monkeypatch `webapp.main.check_groundedness_llm` and the re-embed/re-retrieve call → correct JSON shape.
- No real network calls in these tests — matches the existing `@pytest.mark.slow` convention; real end-to-end exercise of `/api/transcribe` and `/api/reflect` is manual, same as `voice_answer.py` today.
- `webapp/static/*.js` — manually smoke-tested in a browser (no JS test framework), same convention already used for the thin CLI scripts.

## Prerequisites

- Same as Phase 3: `SARVAM_API_KEY`, `ANTHROPIC_API_KEY` in the environment (via `uv run --env-file .env` or exported) for the audio/reflect paths; text-only queries need neither.
- An already-built Qdrant collection (`results/qdrant_local`, already present for all 12 `{hi,en,ta} x {native,sliding_window,sentence,metadata}` combinations from prior phases).
