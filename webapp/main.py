"""The web UI: a thin FastAPI HTTP layer over the same pipeline used
everywhere else (pipeline.run_pipeline, stt.transcribe_audio,
reflect.check_groundedness_llm). Loads Embedder + one Retriever per
available Qdrant collection ONCE at startup (module-level state), reused
across requests -- the same "load once, serve many queries" pattern
scripts/answer.py uses in its input loop."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS
from speculative_rag.embedding import Embedder
from speculative_rag.retrieval import Retriever

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
