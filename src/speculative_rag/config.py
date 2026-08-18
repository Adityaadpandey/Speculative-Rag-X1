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
    # generator actually scores. Matched to answer_top_k: measured on the
    # production en_native collection (392K chunks), the gold passage lands
    # in the top 10 only 71% of the time it's retrievable at all, vs 80% in
    # the top 20 -- and extraction is cheap (single-digit ms regardless of
    # how many short candidate sentences it scores), so capping this below
    # answer_top_k was throwing away already-paid-for retrieval recall for
    # no latency benefit. The LLM fallbacks in answer_orchestrator.py read
    # PipelineResult.retrieved_chunks directly (the full answer_top_k pool),
    # not this value, so raising it doesn't shrink what they see either.
    evidence_top_n: int = 20
    latency_sample_size: int = 200
    # Non-LLM prefilter ahead of the async rerank+regenerate LLM call (see
    # answer_orchestrator.resolve_recovery). Ships disabled: a prior cheap
    # lexical-reranking attempt made recall WORSE, not better (see
    # reflect.py's rerank_passages_llm docstring), so this only flips to True
    # once scripts/run_prefilter_recall_eval.py's offline pass shows zero
    # gold-passage exclusions and the paired LLM eval shows no F1/recall
    # regression -- see the prefilter/eval sections of
    # docs/superpowers/specs/2026-08-18-extraction-quality-and-prefilter-design.md.
    rerank_prefilter_enabled: bool = False
    rerank_prefilter_lexical_top_n: int = 8
    rerank_prefilter_retrieval_reserve: int = 5
    # Async-only dense-cosine rerank over the already-retrieved BM25 pool in
    # answer_orchestrator.resolve_recovery (see dense_rerank.py) -- reorders
    # PipelineResult.retrieved_chunks before the prefilter/LLM-rerank stages,
    # never drops any. Measured on the real en_native production collection
    # (392K chunks): reordering the same top-20 BM25 candidates by dense
    # cosine similarity takes Hit@5-into-the-pool from 0.50 to 0.71. Ships
    # disabled: the dense model (~1GB RSS, see embedding.py) is loaded lazily
    # and only when this flips true, but the free-tier webapp process still
    # can't carry that alongside the sparse hot path without a separate
    # hosting decision (dedicated worker/service, or a smaller dense model)
    # -- flip this on only once that's settled.
    dense_rerank_enabled: bool = False
    # STT + async groundedness check settings.
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_default_language_code: str = "hi-IN"
    stt_latency_sample_size: int = 30
    reflect_latency_sample_size: int = 30


# BCP-47 codes for Sarvam STT, keyed by the language set LANGUAGE_SOURCES uses.
BCP47_LANGUAGE_CODES: dict[str, str] = {
    "hi": "hi-IN",
    "en": "en-IN",
    "ta": "ta-IN",
    "ur": "ur-IN",
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
    # Urdu's standard ISO 639-1 code is "ur" (not "ud").
    LanguageSource(
        language="ur",
        parquet_path=REPO_ROOT / "MSMARCO-XI" / "validation" / "urdval.parquet",
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
