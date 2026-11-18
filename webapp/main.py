"""The web UI: a thin FastAPI HTTP layer over the same pipeline used
everywhere else (pipeline.run_pipeline, stt.transcribe_audio,
openai_reflect.check_groundedness). Loads one SparseEmbedder per language
(BM25's tokenizer/stemmer setup is language-specific, see
sparse_embedding.py) plus one Retriever per available Qdrant collection
once at startup (module-level state), reused across requests -- the same
"load once, serve many queries" pattern scripts/answer.py uses in its
input loop."""

import asyncio
import os
import tempfile
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from speculative_rag.answer_orchestrator import resolve_recovery
from speculative_rag.answer_status import combine_verdict
from speculative_rag.config import BCP47_LANGUAGE_CODES, LANGUAGE_SOURCES, SETTINGS, get_qdrant_client
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.openai_reflect import (
    MODEL as LLM_MODEL,
    check_groundedness,
    rerank_passages,
    stream_generate_answer,
    synthesize_answer,
)
from speculative_rag.reflect import RerankCandidate
from speculative_rag.retrieval import Retriever
from speculative_rag.sparse_embedding import SparseEmbedder
from speculative_rag.stt import SarvamAPIKeyMissingError, transcribe_audio

STATIC_DIR = Path(__file__).resolve().parent / "static"
KNOWN_STRATEGIES = ["native", "sliding_window", "sentence", "metadata"]
KNOWN_LANGUAGES = [source.language for source in LANGUAGE_SOURCES]

_EMBEDDERS: dict[str, SparseEmbedder] = {}
_RETRIEVERS: dict[str, Retriever] = {}
_COLLECTIONS: list[dict[str, str]] = []

# In-memory store for /api/answer's two-phase fast-path/recovery split (see
# post_answer below). Bounded FIFO so a long-running instance doesn't grow
# without limit -- this is a single-process demo/eval webapp, not a queue
# that needs durability across restarts.
_MAX_PENDING_ANSWERS = 500
_PENDING_ANSWERS: OrderedDict[str, dict] = OrderedDict()


def _parse_collection_name(name: str) -> dict[str, str] | None:
    for language in KNOWN_LANGUAGES:
        prefix = f"{language}_"
        if name.startswith(prefix):
            strategy = name[len(prefix):]
            if strategy in KNOWN_STRATEGIES:
                return {"language": language, "strategy": strategy}
    return None


def _load_state() -> None:
    global _EMBEDDERS, _RETRIEVERS, _COLLECTIONS
    _EMBEDDERS = {source.language: SparseEmbedder(source.language) for source in LANGUAGE_SOURCES}
    client = get_qdrant_client()
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
app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")


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
        "openai_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
        # Lets the UI name the model that actually runs recovery instead of
        # a hard-coded label that drifts whenever the provider changes.
        "llm_model": LLM_MODEL,
    }


class QueryRequest(BaseModel):
    query_text: str
    language: str
    strategy: str


@app.post("/api/query", response_model=None)
async def post_query(body: QueryRequest) -> JSONResponse | dict:
    collection_name = f"{body.language}_{body.strategy}"
    retriever = _RETRIEVERS.get(collection_name)
    embedder = _EMBEDDERS.get(body.language)
    if retriever is None or embedder is None:
        return JSONResponse(status_code=404, content={"error": f"No such collection '{collection_name}'"})

    result = run_pipeline(
        body.query_text,
        embedder,
        retriever,
        off_topic_threshold=SETTINGS.off_topic_threshold,
        groundedness_threshold=SETTINGS.groundedness_threshold,
        top_k=SETTINGS.answer_top_k,
        evidence_top_n=SETTINGS.evidence_top_n,
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
        "num_retrieved": result.num_retrieved,
        "num_evidence_candidates": result.num_evidence_candidates,
    }


class AnswerRequest(BaseModel):
    query_text: str
    language: str
    strategy: str


def _serialize_pipeline(pipeline_result) -> dict:
    # Keep this capped at the same pool width used by the answer pipeline:
    # the UI should expose what was actually considered, not only rank 1.
    retrieved_candidates = [
        {
            "rank": rank,
            "score": scored.score,
            "chunk_id": scored.chunk.chunk_id,
            "source_passage_id": scored.chunk.source_passage_id,
            "text": scored.chunk.text,
            "language": scored.chunk.language,
            "query_type": scored.chunk.query_type,
            "strategy": scored.chunk.strategy,
        }
        for rank, scored in enumerate(pipeline_result.retrieved_chunks[:20], start=1)
    ]
    return {
        "outcome": pipeline_result.outcome.value,
        "timings": [{"stage": t.stage, "elapsed_ms": t.elapsed_ms} for t in pipeline_result.timings],
        "total_ms": pipeline_result.total_ms,
        "num_retrieved": pipeline_result.num_retrieved,
        "num_evidence_candidates": pipeline_result.num_evidence_candidates,
        "retrieved_candidates": retrieved_candidates,
    }


def _serialize_outcome(outcome) -> dict:
    return {
        "status": "done",
        "final_status": outcome.final_status.value,
        "answer": (
            {
                "text": outcome.answer_text,
                "source_passage_ids": outcome.source_passage_ids,
                "answer_engine": outcome.answer_engine,
            }
            if outcome.answer_text is not None
            else None
        ),
        "reason": outcome.reason,
        "instrumentation": {
            "initial_answer_valid": outcome.initial_answer_valid,
            "initial_answerable": outcome.initial_answerable,
            "rerank_triggered": outcome.rerank_triggered,
            "rerank_latency_ms": outcome.rerank_latency_ms,
            "selected_source_ids": outcome.selected_source_ids,
            "reranked_answer_generated": outcome.reranked_answer_generated,
            "regenerated_answer_valid": outcome.regenerated_answer_valid,
            "regenerated_answerable": outcome.regenerated_answerable,
            "synthesis_triggered": outcome.synthesis_triggered,
            # Only the recovery phase's own latency -- the fast path is
            # already reported separately in "pipeline" and is what's
            # actually measured against the ~200ms latency budget. This
            # number runs async, after the HTTP response below has already
            # been sent, so it's explicitly not counted against that budget.
            "recovery_latency_ms": outcome.total_end_to_end_latency_ms,
        },
        "pipeline": _serialize_pipeline(outcome.pipeline_result),
    }


def _serialize_recovery_unavailable(pipeline_result, error: Exception) -> dict:
    """Finish polling instead of leaving the UI stuck when Claude is down."""
    answer = pipeline_result.answer
    return {
        "status": "done",
        "final_status": "recovery_unavailable",
        "answer": (
            {
                "text": answer.text,
                "source_passage_ids": [answer.source_passage_id],
                "answer_engine": "extractive",
            }
            if answer is not None
            else None
        ),
        "reason": f"LLM recovery unavailable; showing the fast grounded answer. ({error})",
        "instrumentation": {
            "initial_answer_valid": False,
            "initial_answerable": False,
            "rerank_triggered": False,
            "rerank_latency_ms": None,
            "selected_source_ids": [],
            "reranked_answer_generated": False,
            "regenerated_answer_valid": None,
            "regenerated_answerable": None,
            "synthesis_triggered": False,
            "recovery_latency_ms": 0.0,
        },
        "pipeline": _serialize_pipeline(pipeline_result),
    }


def _stage_reporter(request_id: str):
    # Called from the background thread resolve_recovery runs on (see
    # asyncio.to_thread below) -- plain dict assignment is safe here since
    # each request_id has exactly one writer at a time.
    def on_stage(stage: str) -> None:
        _PENDING_ANSWERS[request_id] = {"status": "resolving", "stage": stage}

    return on_stage


def _answer_reporter(request_id: str):
    partial = [""]

    def on_delta(delta: str) -> None:
        partial[0] += delta
        _PENDING_ANSWERS[request_id] = {
            "status": "resolving",
            "stage": "generating",
            "partial_answer": partial[0],
        }

    return on_delta


async def _run_recovery(request_id: str, query_text: str, pipeline_result) -> None:
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")
        outcome = await asyncio.to_thread(
            resolve_recovery,
            query_text,
            pipeline_result,
            check_groundedness=check_groundedness,
            rerank_passages=rerank_passages,
            synthesize_answer=synthesize_answer,
            on_stage=_stage_reporter(request_id),
            stream_answer=stream_generate_answer,
            on_answer_delta=_answer_reporter(request_id),
            prefilter_enabled=SETTINGS.rerank_prefilter_enabled,
            prefilter_lexical_top_n=SETTINGS.rerank_prefilter_lexical_top_n,
            prefilter_retrieval_reserve=SETTINGS.rerank_prefilter_retrieval_reserve,
            dense_rerank_enabled=SETTINGS.dense_rerank_enabled,
        )
        _PENDING_ANSWERS[request_id] = _serialize_outcome(outcome)
    except Exception as exc:
        # A missing/expired API key or a provider timeout must resolve the
        # request. Otherwise the browser polls "reranking" forever.
        _PENDING_ANSWERS[request_id] = _serialize_recovery_unavailable(pipeline_result, exc)


@app.post("/api/answer", response_model=None)
async def post_answer(body: AnswerRequest, background_tasks: BackgroundTasks) -> JSONResponse | dict:
    """Returns the fast extractive path immediately (this is what's measured
    against the ~200ms latency budget) with a request_id, then -- always, so
    the returned answer is never a stale/unvalidated one shown next to it --
    runs the LLM validate -> rerank -> regenerate -> re-validate ->
    multi-passage-synthesis recovery chain
    (speculative_rag.answer_orchestrator.resolve_recovery) in a background
    thread. Poll GET /api/answer/{request_id} for the validated result once
    it's ready; every LLM call in that chain is a real network round trip
    and is deliberately never allowed to block this response. final_status
    only reads "answered" once the returned answer has actually passed
    validation."""
    collection_name = f"{body.language}_{body.strategy}"
    retriever = _RETRIEVERS.get(collection_name)
    embedder = _EMBEDDERS.get(body.language)
    if retriever is None or embedder is None:
        return JSONResponse(status_code=404, content={"error": f"No such collection '{collection_name}'"})

    pipeline_result = run_pipeline(
        body.query_text,
        embedder,
        retriever,
        off_topic_threshold=SETTINGS.off_topic_threshold,
        groundedness_threshold=SETTINGS.groundedness_threshold,
        top_k=SETTINGS.answer_top_k,
        evidence_top_n=SETTINGS.evidence_top_n,
    )

    request_id = str(uuid.uuid4())
    _PENDING_ANSWERS[request_id] = {"status": "resolving", "stage": "starting"}
    while len(_PENDING_ANSWERS) > _MAX_PENDING_ANSWERS:
        _PENDING_ANSWERS.popitem(last=False)
    background_tasks.add_task(_run_recovery, request_id, body.query_text, pipeline_result)

    fast_answer = None
    if pipeline_result.answer is not None:
        fast_answer = {
            "text": pipeline_result.answer.text,
            "source_passage_ids": [pipeline_result.answer.source_passage_id],
            "answer_engine": "extractive",
        }

    return {
        "request_id": request_id,
        "status": "resolving",
        "fast_answer": fast_answer,
        "pipeline": _serialize_pipeline(pipeline_result),
    }


@app.get("/api/answer/{request_id}", response_model=None)
async def get_answer(request_id: str) -> JSONResponse | dict:
    entry = _PENDING_ANSWERS.get(request_id)
    if entry is None:
        return JSONResponse(status_code=404, content={"error": f"No such request_id '{request_id}'"})
    return entry


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
    embedder = _EMBEDDERS.get(body.language)
    if retriever is None or embedder is None:
        return JSONResponse(status_code=404, content={"error": f"No such collection '{collection_name}'"})

    query_vector = embedder.embed_query(body.query_text)
    retrieval = retriever.search(query_vector, k=1)
    source_text = retrieval.top.chunk.text if retrieval.top is not None else ""

    reflect_start = time.perf_counter()
    verdict = check_groundedness(body.query_text, body.answer_text, source_text)
    reflect_ms = (time.perf_counter() - reflect_start) * 1000.0

    final_status, reason = combine_verdict(RagOutcome.ANSWERED, verdict)
    fallback_triggered = final_status is not None and final_status.value != "answered"

    return {
        "grounded": verdict.grounded,
        "answers_query": verdict.answers_query,
        "complete": verdict.complete,
        "relevant": verdict.relevant,
        "final_status": final_status.value if final_status is not None else None,
        "reason": reason,
        "reasoning": verdict.reasoning,
        "reflect_latency_ms": reflect_ms,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": reason if fallback_triggered else None,
    }


class RerankRequest(BaseModel):
    query_text: str
    language: str
    strategy: str


@app.post("/api/rerank", response_model=None)
async def post_rerank(body: RerankRequest) -> JSONResponse | dict:
    collection_name = f"{body.language}_{body.strategy}"
    retriever = _RETRIEVERS.get(collection_name)
    embedder = _EMBEDDERS.get(body.language)
    if retriever is None or embedder is None:
        return JSONResponse(status_code=404, content={"error": f"No such collection '{collection_name}'"})

    query_vector = embedder.embed_query(body.query_text)
    retrieval = retriever.search(query_vector, k=SETTINGS.answer_top_k)
    candidates = retrieval.scored_chunks
    if not candidates:
        return JSONResponse(status_code=404, content={"error": "no candidates retrieved for rerank"})

    by_chunk_id = {c.chunk.chunk_id: c for c in candidates}
    rerank_start = time.perf_counter()
    verdict = rerank_passages(
        body.query_text, [RerankCandidate(source_id=c.chunk.chunk_id, text=c.chunk.text) for c in candidates]
    )
    rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

    selected_sources = [
        {
            "source_id": s.source_id,
            "source_passage_id": by_chunk_id[s.source_id].chunk.source_passage_id,
            "relevance_score": s.relevance_score,
        }
        for s in verdict.selected_sources
        if s.source_id in by_chunk_id
    ]

    return {
        "selected_sources": selected_sources,
        "answerable": verdict.answerable,
        "reasoning": verdict.reasoning,
        "rerank_latency_ms": rerank_ms,
    }


class SynthesizeRequest(BaseModel):
    query_text: str
    language: str
    strategy: str


@app.post("/api/synthesize", response_model=None)
async def post_synthesize(body: SynthesizeRequest) -> JSONResponse | dict:
    collection_name = f"{body.language}_{body.strategy}"
    retriever = _RETRIEVERS.get(collection_name)
    embedder = _EMBEDDERS.get(body.language)
    if retriever is None or embedder is None:
        return JSONResponse(status_code=404, content={"error": f"No such collection '{collection_name}'"})

    query_vector = embedder.embed_query(body.query_text)
    retrieval = retriever.search(query_vector, k=SETTINGS.answer_top_k)
    candidates = retrieval.scored_chunks
    if not candidates:
        return JSONResponse(status_code=404, content={"error": "no candidates retrieved for synthesis"})

    synth_start = time.perf_counter()
    verdict = synthesize_answer(body.query_text, [c.chunk.text for c in candidates])
    synth_ms = (time.perf_counter() - synth_start) * 1000.0

    sources = [
        {"source_passage_id": candidates[i].chunk.source_passage_id, "retrieval_score": candidates[i].score}
        for i in verdict.source_indices
    ]

    return {
        "answerable": verdict.answerable,
        "answer_text": verdict.answer_text,
        "sources": sources,
        "reasoning": verdict.reasoning,
        "synthesis_latency_ms": synth_ms,
    }
