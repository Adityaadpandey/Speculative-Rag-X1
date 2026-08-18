# Speculative RAG — Voice-Enabled Indic RAG

A retrieval-augmented QA system over [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) (MS MARCO machine-translated into 14 Indic languages), scoped to Hindi, English, and Tamil. Voice query → Sarvam STT → BM25 sparse retrieval (Qdrant) → extractive answer, all inside a **<200ms hot-path SLA**, with optional async LLM enhancements (groundedness check, rerank, multi-passage synthesis) layered on top and reported separately.

Full design rationale — language scoping, STT provider choice, the latency budget breakdown, chunking-strategy comparison, and the guardrail split — is in [`spec.md`](spec.md).

## How it works

```
voice/text query
      │
      ▼
 STT (Sarvam)                    outside the 200ms SLA, timed separately
      │
      ▼
┌─── hot path (<200ms), no model calls ───────────────────┐
│ unsafe-input check → BM25 embed → Qdrant retrieve        │
│   → off-topic check → extractive answer selection        │
└────────────────────────────────────────────────────────┘
      │
      ▼
 ANSWERED / OFF_TOPIC_REFUSED / UNSAFE_INPUT_REFUSED / INSUFFICIENT_EVIDENCE
      │
      ▼ (async, only if the hot-path answer needs it)
 LLM groundedness check → rerank → multi-passage synthesis
 (Claude Sonnet 5; logged as its own latency, never counted against the SLA)
```

- **Retrieval** is BM25 sparse term-frequency search (`sparse_embedding.py`), not a dense neural embedder in-process — deliberate, so the deployed webapp fits a free-tier 512MB host.
- **The hot path never calls a hosted LLM.** Answer generation is extractive (rank/select a passage or span), which is what makes the <200ms target hittable under real network conditions.
- **Async recovery** (`answer_orchestrator.py`) is the single control-flow owner for what happens when the fast answer fails validation: rerank → regenerate → multi-passage synthesis, each step logged with its own `FinalAnswerStatus`.

## Project layout

| Path | Purpose |
|---|---|
| `src/speculative_rag/` | Core library: pipeline, retrieval, chunking, guardrails, answer orchestration, STT, async LLM reflect/rerank/synthesis |
| `webapp/` | FastAPI app (`main.py`) + static frontend (`static/`) serving the pipeline over HTTP |
| `scripts/` | CLIs for ingestion, interactive querying, and eval (recall@k, latency, answer quality, embedding-model A/B, rerank/synthesis recovery) |
| `tests/` | pytest suite covering every module above |
| `results/` | Eval outputs (recall@k per chunking strategy, latency percentiles, answer-quality reports) |
| `spec.md` | The technical plan this project implements |

## Setup

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` in the repo root (loaded automatically by `config.py`):

```
SARVAM_API_KEY=       # speech-to-text
ANTHROPIC_API_KEY=    # async reflect/rerank/synthesis (Claude Sonnet 5)
QDRANT_URL=           # unset -> falls back to local on-disk Qdrant at results/qdrant_local
QDRANT_API_KEY=
```

The `ai4bharat/MSMARCO-XI` dataset is not vendored — it's pulled separately (its own git/HF repo) into `MSMARCO-XI/` and is gitignored.

## Running

```bash
# Web UI (text + voice query, full pipeline)
uv run python -m scripts.serve_ui

# Interactive extractive-answer CLI over an existing collection
uv run python -m scripts.answer

# Voice-in CLI: audio file -> transcript -> pipeline answer
uv run python -m scripts.voice_answer path/to/audio.wav
```

Optional dense-embedding extra (only needed for `run_embedding_model_eval.py` and dense-embedder tests — the deployed app doesn't use it):

```bash
uv sync --extra dense
```

## Evaluation

Each script in `scripts/run_*_eval.py` reproduces one table in `results/`:

- `run_recall_eval.py` — recall@k per chunking strategy vs. the dataset's gold `is_selected` labels (`chunking_comparison.md`)
- `run_latency_eval.py` / `run_full_latency_eval.py` — P50/P70/P100 latency, RAG-pipeline-only and the three-way STT/RAG/async-check split (`latency_rag.md`)
- `run_answer_quality_eval.py` — end-to-end answer quality against gold answers, per language
- `run_embedding_model_eval.py` — dense embedding model A/B (requires the `dense` extra)
- `run_llm_rerank_eval.py` / `run_synthesis_eval.py` — whether the async LLM recovery steps actually improve on the hot-path answer

## Testing

```bash
uv run pytest
```

## Deployment

`Dockerfile` + `render.yaml` deploy the webapp to Render's free tier (BM25-only, no in-process dense embedder, sized for 512MB RAM). Requires a separate Qdrant Cloud cluster — set `QDRANT_URL`/`QDRANT_API_KEY` in the Render dashboard; nothing is provisioned automatically.
