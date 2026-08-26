"""HTTP generator target: calls the real webapp over HTTP instead of
importing production code in-process. Points at LOCAL_WEBAPP_URL (default
http://127.0.0.1:8000), not the deployed Render instance -- calling the
deployed instance from a dev machine adds a real ~175ms hop to Qdrant Cloud
each way that the deployed instance itself doesn't pay (it's co-located
with the Qdrant cluster; this dev machine isn't).

Unlike app.generator, this ignores the harness's own `results` argument
(the eval loop's isolated FAISS-search hits) entirely -- the whole point of
going over HTTP is that the target's real /api/query endpoint does its own
real BM25/Qdrant retrieval against the real production corpus, which is
what should feed generation here, not the harness's synthetic candidates.

Calls /api/query (the synchronous fast path) rather than /api/answer,
matching this suite's target project's own convention of only measuring
the guaranteed-synchronous hot path -- webapp/main.py's LLM recovery chain
runs async, background-only, specifically outside that budget.
"""
import os
import time

import httpx

from speculative_rag.pipeline_schemas import RagOutcome

BASE_URL = os.environ.get("LOCAL_WEBAPP_URL", "http://127.0.0.1:8000")
LANGUAGE = os.environ.get("EVAL_HTTP_LANGUAGE", "en")
STRATEGY = os.environ.get("EVAL_HTTP_STRATEGY", "native")

GENERATION_BACKEND = "http"
GENERATION_MODEL = "webapp-http"


class GeneratedAnswer:
    def __init__(self, text: str, grounded: bool, generation_ms: float, model: str):
        self.text = text
        self.grounded = grounded
        self.generation_ms = generation_ms
        self.model = model


_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(base_url=BASE_URL, timeout=30.0)
    return _client


def generate_answer(query: str, results: list) -> GeneratedAnswer:
    client = _get_client()
    t0 = time.perf_counter()
    response = client.post(
        "/api/query",
        json={"query_text": query, "language": LANGUAGE, "strategy": STRATEGY},
    )
    generation_ms = (time.perf_counter() - t0) * 1000.0
    response.raise_for_status()
    body = response.json()

    answered = body.get("outcome") == RagOutcome.ANSWERED.value and body.get("answer") is not None
    text = body["answer"]["text"] if answered else ""
    return GeneratedAnswer(text=text, grounded=answered, generation_ms=generation_ms, model=GENERATION_MODEL)
