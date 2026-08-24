"""OPTIONAL config surface rag-local-eval-loop reads defensively (see
TARGET_INTERFACE.md) -- also loads the project's own .env so OPENAI_API_KEY
(or ANTHROPIC_API_KEY) set there reaches eval/judge.py, matching
speculative_rag.config's own load_dotenv call.
"""
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

GENERATION_BACKEND = "recovery-chain"  # network-bound (real Anthropic calls) -- not "local", so workers stay unclamped
GENERATION_MODEL = "speculative-rag-answer-orchestrator"
