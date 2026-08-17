from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

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
    off_topic_threshold: float = 0.75
    groundedness_threshold: float = 0.3
    default_language: str = "hi"
    default_strategy: str = "native"
    answer_top_k: int = 5
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
    ),
    LanguageSource(
        language="ta",
        parquet_path=REPO_ROOT / "MSMARCO-XI" / "validation" / "tamval.parquet",
        text_field="Translated_passages",
    ),
]

SETTINGS = Settings()
