import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from qdrant_client import QdrantClient

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every entrypoint (serve_ui, answer, voice_answer, eval scripts) imports this
# module first (directly or via SETTINGS), so loading it here guarantees
# SARVAM_API_KEY/ANTHROPIC_API_KEY are in os.environ before anything reads
# them, regardless of which script runs or what the cwd is. Already-exported
# env vars still win (load_dotenv defaults to override=False).
load_dotenv(REPO_ROOT / ".env")


class LanguageSource(BaseModel):
    language: str
    parquet_path: Path
    text_field: str  # "Translated_passages" or "English_passages"
    query_field: str = "query"  # "query" (native-language) or "Eng_Query"


class Settings(BaseModel):
    sample_size: int = 300
    sample_seed: int = 42
    embedding_model: str = "intfloat/multilingual-e5-small"
    qdrant_path: Path = REPO_ROOT / "results" / "qdrant_local"
    results_dir: Path = REPO_ROOT / "results"
    sliding_window_words: int = 40
    sliding_window_overlap: int = 10
    sentence_max_words: int = 40
    sentence_overlap_sentences: int = 1
    top_k_values: list[int] = [1, 5, 10]
    # Hot-path pipeline settings.
    # BM25 sparse scores are unbounded (no fixed max) and a positive score
    # doesn't imply topical relevance -- see guardrails.is_off_topic. This is
    # a near-zero floor ("shares at least one real term with the corpus"),
    # not a relevance cutoff like the old dense-cosine 0.75 threshold was.
    off_topic_threshold: float = 1e-6
    groundedness_threshold: float = 0.3
    default_language: str = "hi"
    default_strategy: str = "native"
    # Retrieval pool width: how many BM25 candidates come back from Qdrant.
    # Wider than the number actually used for extractive generation
    # (evidence_top_n) so the hot-path evidence-selection stage and the LLM
    # rerank/synthesis fallbacks all draw from the same, larger candidate set
    # instead of only ever seeing embedding rank-1.
    answer_top_k: int = 20
    # How many of those retrieved candidates the hot-path extractive
    # generator actually scores. Kept smaller than answer_top_k so the fast
    # path stays cheap while the LLM fallbacks still get the wider pool.
    evidence_top_n: int = 5
    latency_sample_size: int = 200
    # STT + async groundedness check settings.
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_default_language_code: str = "hi-IN"
    stt_latency_sample_size: int = 30
    reflect_latency_sample_size: int = 30


# BCP-47 codes for Sarvam STT, keyed by the {hi, en, ta} set LANGUAGE_SOURCES uses.
BCP47_LANGUAGE_CODES: dict[str, str] = {
    "hi": "hi-IN",
    "en": "en-IN",
    "ta": "ta-IN",
}

LANGUAGE_SOURCES: list[LanguageSource] = [
    LanguageSource(
        language="hi",
        parquet_path=REPO_ROOT / "MSMARCO-XI" / "validation" / "hinval.parquet",
        text_field="Translated_passages",
    ),
    LanguageSource(
        language="en",
        parquet_path=REPO_ROOT / "MSMARCO-XI" / "validation" / "hinval.parquet",
        text_field="English_passages",
        query_field="Eng_Query",
    ),
    LanguageSource(
        language="ta",
        parquet_path=REPO_ROOT / "MSMARCO-XI" / "validation" / "tamval.parquet",
        text_field="Translated_passages",
    ),
]

SETTINGS = Settings()


def get_qdrant_client() -> QdrantClient:
    """QDRANT_URL set (e.g. a Qdrant Cloud cluster) -> remote client, for the
    deployed webapp and production ingestion. Unset -> local file-based
    client at SETTINGS.qdrant_path, for local dev/eval scripts -- keeps
    every existing script working unchanged with no env setup required.

    The remote client gets an explicit, generous timeout: the qdrant-client
    default is a few seconds, which a free-tier cloud cluster can't reliably
    meet for a multi-thousand-point upsert with full sparse vectors + text
    payloads (bulk ingestion into the eu-central-1 free tier was timing out
    well under 60s otherwise). Cheap webapp search calls stay fast anyway --
    a longer timeout only matters when a request is actually slow.

    Deliberately not prefer_grpc=True: I tried it and measured it head-to-
    head against this cluster (10 warm calls each) -- REST was a flat
    ~175ms every time, gRPC alternated between ~175ms and ~350ms, adding a
    recurring tail-latency spike (p95/p99) REST doesn't have. The slow
    retrieve I was originally chasing turned out to be a one-off
    cold-connection cost on the first request after process startup, not a
    REST-vs-gRPC difference -- REST warm is already at this network's floor
    against Qdrant Cloud."""
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(
            url=url,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=60,
        )
    return QdrantClient(path=str(SETTINGS.qdrant_path))
