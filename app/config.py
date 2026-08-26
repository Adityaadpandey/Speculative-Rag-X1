"""OPTIONAL config surface rag-local-eval-loop reads defensively (see
TARGET_INTERFACE.md) -- also loads the project's own .env so OPENAI_API_KEY
(or ANTHROPIC_API_KEY) set there reaches eval/judge.py, matching
speculative_rag.config's own load_dotenv call.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# The evaluator measures the synchronous target call. Production LLM
# recovery is intentionally background-only in webapp/main.py. Set
# EVAL_GENERATION_MODE=recovery (or use eval.runner --generation-mode
# recovery) only for the deliberate off-SLA recovery comparison run.
EVAL_GENERATION_MODE = os.environ.get("EVAL_GENERATION_MODE", "local").lower()
if EVAL_GENERATION_MODE not in {"local", "recovery"}:
    raise ValueError("EVAL_GENERATION_MODE must be 'local' or 'recovery'")
EVAL_RECOVERY_PROVIDER = os.environ.get("EVAL_RECOVERY_PROVIDER", "anthropic").lower()
if EVAL_RECOVERY_PROVIDER not in {"anthropic", "openai"}:
    raise ValueError("EVAL_RECOVERY_PROVIDER must be 'anthropic' or 'openai'")

GENERATION_BACKEND = "recovery-chain" if EVAL_GENERATION_MODE == "recovery" else "local"
LOCAL_GENERATION_MODEL = "extractive"
GENERATION_MODEL = (
    "claude-haiku-4-5-20251001"
    if EVAL_GENERATION_MODE == "recovery" and EVAL_RECOVERY_PROVIDER == "anthropic"
    else "openai-recovery-chain"
    if EVAL_GENERATION_MODE == "recovery"
    else "extractive"
)
LATENCY_BUDGET_MS = 50.0
GENERATION_LATENCY_TARGET_MS = 200.0
