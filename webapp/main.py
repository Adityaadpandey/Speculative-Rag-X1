"""The web UI: a thin FastAPI HTTP layer over the same pipeline used
everywhere else (pipeline.run_pipeline, stt.transcribe_audio,
reflect.check_groundedness_llm). Loads Embedder + one Retriever per
available Qdrant collection ONCE at startup (module-level state), reused
across requests -- the same "load once, serve many queries" pattern
scripts/answer.py uses in its input loop."""

import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient

from speculative_rag.config import BCP47_LANGUAGE_CODES, LANGUAGE_SOURCES, SETTINGS
from speculative_rag.embedding import Embedder
from speculative_rag.generation import select_answer_span
from speculative_rag.pipeline import run_pipeline
from speculative_rag.reflect import check_groundedness_llm, select_best_candidate_llm
from speculative_rag.retrieval import Retriever
from speculative_rag.stt import SarvamAPIKeyMissingError, transcribe_audio

STATIC_DIR = Path(__file__).resolve().parent / "static"
KNOWN_STRATEGIES = ["native", "sliding_window", "sentence", "metadata"]
KNOWN_LANGUAGES = [source.language for source in LANGUAGE_SOURCES]

_EMBEDDER: Embedder | None = None
_RETRIEVERS: dict[str, Retriever] = {}
_COLLECTIONS: list[dict[str, str]] = []


def _parse_collection_name(name: str) -> dict[str, str] | None:
    for language in KNOWN_LANGUAGES:
        prefix = f"{language}_"
        if name.startswith(prefix):
            strategy = name[len(prefix):]
            if strategy in KNOWN_STRATEGIES:
                return {"language": language, "strategy": strategy}
    return None


def _load_state() -> None:
    global _EMBEDDER, _RETRIEVERS, _COLLECTIONS
    _EMBEDDER = Embedder()
    client = QdrantClient(path=str(SETTINGS.qdrant_path))
    retrievers: dict[str, Retriever] = {}
    collections: list[dict[str, str]] = []
    for collection in client.get_collections().collections:
        parsed = _parse_collection_name(collection.name)
        if parsed is None:
            continue
        retrievers[collection.name] = Retriever(client, collection.name)
        collections.append(parsed)
    _RETRIEVERS = retrievers
    _COLLECTIONS = collections


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/collections")
async def get_collections() -> dict:
    return {
        "collections": _COLLECTIONS,
        "sarvam_key_configured": bool(os.environ.get("SARVAM_API_KEY")),
        "anthropic_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


class QueryRequest(BaseModel):
    query_text: str
    language: str
    strategy: str


@app.post("/api/query", response_model=None)
async def post_query(body: QueryRequest) -> JSONResponse | dict:
    collection_name = f"{body.language}_{body.strategy}"
    retriever = _RETRIEVERS.get(collection_name)
    if retriever is None:
        return JSONResponse(status_code=404, content={"error": f"No such collection '{collection_name}'"})

    result = run_pipeline(
        body.query_text,
        _EMBEDDER,
        retriever,
        off_topic_threshold=SETTINGS.off_topic_threshold,
        groundedness_threshold=SETTINGS.groundedness_threshold,
        top_k=SETTINGS.answer_top_k,
    )

    answer = None
    if result.answer is not None:
        answer = {
            "text": result.answer.text,
            "source_passage_id": result.answer.source_passage_id,
            "retrieval_score": result.answer.retrieval_score,
            "groundedness_score": result.answer.groundedness_score,
        }

    return {
        "outcome": result.outcome.value,
        "answer": answer,
        "refusal_reason": result.refusal_reason,
        "timings": [{"stage": t.stage, "elapsed_ms": t.elapsed_ms} for t in result.timings],
        "total_ms": result.total_ms,
    }


@app.post("/api/transcribe", response_model=None)
async def post_transcribe(file: UploadFile = File(...), language: str = Form(...)) -> JSONResponse | dict:
    language_code = BCP47_LANGUAGE_CODES.get(language, SETTINGS.sarvam_default_language_code)
    suffix = Path(file.filename or "").suffix or ".wav"
    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        stt_start = time.perf_counter()
        transcript = transcribe_audio(tmp_path, language_code)
        stt_ms = (time.perf_counter() - stt_start) * 1000.0
    except SarvamAPIKeyMissingError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"transcript": transcript, "stt_latency_ms": stt_ms}


class ReflectRequest(BaseModel):
    query_text: str
    language: str
    strategy: str
    answer_text: str


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

    return {
        "grounded": verdict.grounded,
        "answers_query": verdict.answers_query,
        "reasoning": verdict.reasoning,
        "reflect_latency_ms": reflect_ms,
    }


class RerankRequest(BaseModel):
    query_text: str
    language: str
    strategy: str


@app.post("/api/rerank", response_model=None)
async def post_rerank(body: RerankRequest) -> JSONResponse | dict:
    collection_name = f"{body.language}_{body.strategy}"
    retriever = _RETRIEVERS.get(collection_name)
    if retriever is None:
        return JSONResponse(status_code=404, content={"error": f"No such collection '{collection_name}'"})

    query_vector = _EMBEDDER.embed_query(body.query_text)
    retrieval = retriever.search(query_vector, k=SETTINGS.answer_top_k)
    candidates = retrieval.scored_chunks
    if not candidates:
        return JSONResponse(status_code=404, content={"error": "no candidates retrieved for rerank"})

    rerank_start = time.perf_counter()
    verdict = select_best_candidate_llm(body.query_text, [c.chunk.text for c in candidates])
    rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

    chosen = candidates[verdict.best_index]
    span = select_answer_span(body.query_text, chosen.chunk.text)

    return {
        "text": span,
        "source_passage_id": chosen.chunk.source_passage_id,
        "retrieval_score": chosen.score,
        "reasoning": verdict.reasoning,
        "rerank_latency_ms": rerank_ms,
    }
