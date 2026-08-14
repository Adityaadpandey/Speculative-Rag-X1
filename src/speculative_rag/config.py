from pathlib import Path

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]


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
