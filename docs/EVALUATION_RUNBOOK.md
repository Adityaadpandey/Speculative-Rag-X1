# Evaluation Runbook

Repeatable wiring process for attaching `rag-local-eval-loop` to a cloned Task #2 submission and measuring the values it actually produces.

## Prerequisites

- A cloned submission repository.
- Python 3.10+ for the evaluator.
- An `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for the evaluator's independent judge.
- The submission's own service or native Python dependencies available.

## 1. Clone the evaluator once

Reuse one clone for every submission:

```bash
git clone https://github.com/BeaconBandhu/rag-local-eval-loop.git /path/to/rag-local-eval-loop
```

On Windows, for example:

```powershell
git clone https://github.com/BeaconBandhu/rag-local-eval-loop.git C:\tools\rag-local-eval-loop
```

## 2. Attach it to a submission

Copy the evaluator's `eval/` directory and launcher next to the submission's files:

```powershell
$sub = "C:\evals\task2-submissions\<team-name>"
Copy-Item C:\tools\rag-local-eval-loop\eval "$sub\eval" -Recurse
Copy-Item C:\tools\rag-local-eval-loop\run.ps1 "$sub\run.ps1"
```

On macOS/Linux:

```bash
cp -R /path/to/rag-local-eval-loop/eval /path/to/submission/eval
cp /path/to/rag-local-eval-loop/run.sh /path/to/submission/run.sh
```

Alternatively, leave the submission untouched and pass `--rag-root /path/to/submission` on every evaluator run.

This repository already has the attached evaluator under `eval/`, plus `run.ps1` and `run.sh`.

The report linked from the README was produced by `eval.runner`, not one of the older `scripts/run_*_eval.py` tables. Its generation path is selected by `--generation-mode`:

- `local` (default): `app.generator` stops after the zero-LLM extractive path.
- `recovery`: `app.generator` calls `answer_orchestrator.resolve_recovery` synchronously, using the existing `reflect.py` Claude Haiku functions. If the Anthropic account has no credits, pass `--recovery-provider openai` to use the same chain with the project's OpenAI adapter.

## 3. Create the evaluator environment

The evaluator uses the submission's `.venv`:

```bash
cd /path/to/submission
python3 -m venv .venv
.venv/bin/python -m pip install -r /path/to/rag-local-eval-loop/requirements.txt
```

On Windows:

```powershell
cd $sub
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r C:\tools\rag-local-eval-loop\requirements.txt
```

For a native Python target, install the submission's own dependencies into this same environment as well.

## 4. Configure the target interface

### Native Python

The default interface is:

```text
app.embedder.embed()
app.embedder.embed_one()
app.embedder.get_model()
app.generator.generate_answer()
```

For different module names, set:

```powershell
$env:EVAL_EMBEDDER_MODULE = "main"
$env:EVAL_GENERATOR_MODULE = "main"
```

### HTTP service

Use `eval.http_target` and configure the service's real endpoints with a JSON file. Copy an example config from the evaluator repository, then set:

```powershell
$env:EVAL_EMBEDDER_MODULE = "eval.http_target"
$env:EVAL_GENERATOR_MODULE = "eval.http_target"
$env:EVAL_HTTP_CONFIG = "C:\path\to\target-config.json"
```

`eval/http_target.py` contains the complete config schema and validation errors.

## 5. Configure the judge

The judge key is independent of any key used by the submission:

```powershell
$env:OPENAI_API_KEY = "sk-..."
# or:
$env:ANTHROPIC_API_KEY = "..."
```

Without a judge credential, faithfulness and correctness are reported as skipped.

## 6. Smoke test

Run a small sample before spending judge calls:

```powershell
.\run.ps1 --num-answerable 3 --num-unanswerable 3 --workers 1
```

macOS/Linux:

```bash
./run.sh --num-answerable 3 --num-unanswerable 3 --workers 1
```

A wiring failure should identify the missing module, function, or endpoint. Fix that specific issue and rerun the smoke test.

## 7. Full evaluation

```powershell
.\run.ps1 --num-answerable 50 --num-unanswerable 50 --workers 1 --judge-workers 8
```

macOS/Linux:

```bash
TOKENIZERS_PARALLELISM=false ./run.sh \
  --num-answerable 50 --num-unanswerable 50 \
  --workers 1 --judge-workers 8
```

The report is printed to the console and saved as `results/<timestamp>.json`.

### Full recovery comparison

Run the exact same deterministic sample through the synchronous LLM recovery chain. Keep `--language hin`, `--split validation`, `--num-answerable 50`, `--num-unanswerable 50`, `--seed 42`, and `--top-k 5` unchanged:

```bash
TOKENIZERS_PARALLELISM=false ./run.sh \
  --generation-mode recovery \
  --recovery-provider openai \
  --report-suffix _recovery \
  --num-answerable 50 --num-unanswerable 50 \
  --workers 1 --judge-workers 8
```

PowerShell:

```powershell
.\run.ps1 `
  --generation-mode recovery `
  --recovery-provider openai `
  --report-suffix _recovery `
  --num-answerable 50 --num-unanswerable 50 `
  --workers 1 --judge-workers 8
```

This produces `results/<timestamp>_recovery.json` without overwriting the local hot-path report. Recovery latency is expected to exceed the 200 ms hot-path SLA because this comparison intentionally waits for Claude.

## What the checks mean

- **Retrieval:** Recall@1/3/5 and MRR against MSMARCO-XI's selected passage labels.
- **Faithfulness:** hallucination rate judged only against the retrieved context, never the gold answer.
- **Correctness:** answer quality judged against MSMARCO-XI's reference answer.
- **Reliability:** false refusals and false confidence on answerable/unanswerable rows; false confidence is the key guardrail metric.
- **Latency:** real embedding, search, retrieval-total, and generation percentiles. This project declares a 200 ms generation target and a 50 ms retrieval target.

## Latest local evaluation

Report: [`results/20260825T054515Z.json`](../results/20260825T054515Z.json)

- Retrieval Recall@5: **90%**
- Faithful rate: **92.31%**
- Correctness: **57.14%**
- False refusal: **30%**
- False confidence: **34%**
- Retrieval P95: **13.67 ms**
- Generation P95: **0.76 ms**
- Latency targets: **PASS**

### Recovery comparison (same 100 queries)

Report: [`results/20260826T111618Z_recovery.json`](../results/20260826T111618Z_recovery.json)

Same `hin` validation sample, same seed, same 50/50 split, but with every answer routed through the synchronous recovery chain (`EVAL_RECOVERY_PROVIDER=openai`, since the Anthropic account was out of credits):

| metric | local (extractive) | recovery (LLM chain) | delta |
|---|---:|---:|---:|
| correctness.correct_rate | 0.571 | 0.795 | **+22.4pp** |
| reliability.false_confidence_rate | 0.340 | 0.660 | +32.0pp (worse) |
| reliability.false_refusal_rate | 0.300 | 0.120 | **-18.0pp (better)** |
| faithfulness.faithful_rate | 0.923 | 0.883 | -4.0pp |
| latency.generation P95 | 0.76 ms | 23752.60 ms | off-SLA by design |
| latency.generation P99 | 0.78 ms | 26384.85 ms | off-SLA by design |

The recovery chain measurably improves correctness (+22pp) and reduces false refusals (-18pp), but at the cost of a sharply worse false-confidence rate (+32pp): the LLM chain confidently answers many unanswerable queries. Its latency is intentionally outside the 200 ms hot-path SLA (it waits for real Claude calls); the web app keeps this work in a background task.
