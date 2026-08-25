"""OPTIONAL config surface rag-local-eval-loop reads defensively (see
TARGET_INTERFACE.md) -- also loads the project's own .env so OPENAI_API_KEY
(or ANTHROPIC_API_KEY) set there reaches eval/judge.py, matching
speculative_rag.config's own load_dotenv call.
"""
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# The evaluator measures the synchronous target call. Production LLM
# recovery is intentionally background-only in webapp/main.py.
GENERATION_BACKEND = "local"
LOCAL_GENERATION_MODEL = "extractive"
GENERATION_MODEL = "extractive"
LATENCY_BUDGET_MS = 50.0
GENERATION_LATENCY_TARGET_MS = 200.0
