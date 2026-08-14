# Phase 1: Data Prep, Chunking Strategies & Recall@K Retrieval Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working, testable pipeline that loads the scoped MSMARCO-XI language subset, dedupes passages, chunks them with 3-4 comparable strategies, indexes each strategy's chunks into Qdrant with a small multilingual embedding model, and reports real Recall@K numbers per strategy per language against the dataset's gold `is_selected` labels.

**Architecture:** A small `src/speculative_rag` package with pure-logic modules (schemas, data loading/dedup, chunking, recall computation) that are unit-testable without any model/network dependency, plus two thin integration modules (embedding, Qdrant indexing) that wrap external libraries. A CLI script wires them together end-to-end and writes a results table to `results/`.

**Tech Stack:** Python 3.13, `uv` for dependency/venv management, `pandas`/`pyarrow` for parquet I/O, `pydantic` for schemas, `sentence-transformers` (`intfloat/multilingual-e5-small`) for embeddings, `qdrant-client` in local on-disk mode (no Docker required), `pytest`.

## Global Constraints

- Python `>=3.13` (already pinned in `.python-version` and `pyproject.toml`).
- Use `uv add` / `uv run` for all dependency and execution commands — this project is a `uv`-managed project (`pyproject.toml` + `.python-version`, no `requirements.txt`).
- Scope to 3 language views per spec §1: **Hindi** (`Translated_passages` from `hinval.parquet`), **English** (`English_passages` from the same `hinval.parquet` rows — same underlying MS MARCO passages, proves the pipeline isn't Hindi-hardcoded), **Tamil** (`Translated_passages` from `tamval.parquet`, script-different proof point).
- Use the **validation** split only, never `train` (spec §7: keep eval cleanly separated from anything used to tune choices; also `train` parquet files are 3.5GB+ each — far too large for iterative dev).
- Sample size: 300 queries per language, seeded (`seed=42`) for reproducibility, drawn only from rows with at least one `is_selected == 1` passage (rows with zero gold passages can't contribute to Recall@K and must be filtered before sampling).
- Dedup passages by `(language, text)` before indexing — never index one point per `(query, passage)` pair (spec §1).
- Chunking strategies operate **within a single passage**, never across passages — MS MARCO candidate passages for one query come from unrelated source documents, so concatenating them across passages would fabricate adjacency that doesn't exist. This is a deliberate, documented deviation from a literal reading of spec §4's sliding-window/sentence description; see Task 5 notes.
- No hosted LLM API calls anywhere in this phase — this phase is retrieval-only.
- Embedding model: `intfloat/multilingual-e5-small` — small (118M params), multilingual, covers Hindi/English/Tamil, follows the `"query: "` / `"passage: "` prefix convention required by E5 models.
- Vector store: `qdrant-client` local on-disk mode (`QdrantClient(path=...)`) — avoids a Docker dependency for this phase while still exercising real Qdrant indexing/search code paths.

---

## File Structure

- `pyproject.toml` — add runtime deps (`pandas`, `pyarrow`, `pydantic`, `sentence-transformers`, `qdrant-client`) and dev dep (`pytest`).
- `src/speculative_rag/__init__.py` — empty package marker.
- `src/speculative_rag/config.py` — `Settings`, `LanguageSource`, `LANGUAGE_SOURCES`, `SETTINGS`.
- `src/speculative_rag/schemas.py` — `Passage`, `Chunk`, `QueryRecord` pydantic models.
- `src/speculative_rag/data.py` — parquet loading, sampling/filtering, passage dedup, `QueryRecord` construction.
- `src/speculative_rag/chunking.py` — `chunk_native`, `chunk_sliding_window`, `chunk_sentence_aware`, `chunk_metadata_aware`, `CHUNKERS` registry.
- `src/speculative_rag/embedding.py` — `Embedder` wrapping `SentenceTransformer`.
- `src/speculative_rag/indexing.py` — `ChunkIndex` wrapping a `QdrantClient` collection.
- `src/speculative_rag/eval.py` — `recall_hit`, `compute_recall_at_k`.
- `scripts/run_recall_eval.py` — CLI entrypoint: loads all language sources, runs all strategies, prints + saves the comparison table.
- `tests/test_data.py`, `tests/test_chunking.py`, `tests/test_eval.py` — pure-logic unit tests, no network/model needed.
- `tests/test_embedding.py`, `tests/test_indexing.py` — integration tests, marked `@pytest.mark.slow` (download the real model / spin up a temp local Qdrant).
- `results/` — output directory for the Recall@K comparison table (gitignored except the final markdown report, see Task 10).

---

### Task 1: Project scaffolding & dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `src/speculative_rag/__init__.py`
- Create: `src/speculative_rag/config.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: `Settings` (pydantic `BaseModel`) with fields `sample_size: int`, `sample_seed: int`, `embedding_model: str`, `qdrant_path: Path`, `results_dir: Path`, `sliding_window_words: int`, `sliding_window_overlap: int`, `sentence_max_words: int`, `sentence_overlap_sentences: int`, `top_k_values: list[int]`.
- Produces: `LanguageSource` (pydantic `BaseModel`) with fields `language: str`, `parquet_path: Path`, `text_field: str`.
- Produces: `LANGUAGE_SOURCES: list[LanguageSource]`, `SETTINGS: Settings` module-level constants.

- [ ] **Step 1: Add dependencies via uv**

```bash
uv add pandas pyarrow pydantic sentence-transformers qdrant-client
uv add --dev pytest
```

- [ ] **Step 2: Create package directories**

```bash
mkdir -p src/speculative_rag tests scripts results
touch src/speculative_rag/__init__.py tests/__init__.py
```

- [ ] **Step 3: Write `src/speculative_rag/config.py`**

```python
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
```

- [ ] **Step 4: Verify it imports cleanly**

Run: `uv run python -c "from speculative_rag.config import SETTINGS, LANGUAGE_SOURCES; print(SETTINGS.embedding_model); print([l.language for l in LANGUAGE_SOURCES])"`
Expected: prints `intfloat/multilingual-e5-small` then `['hi', 'en', 'ta']`. If it fails with `ModuleNotFoundError: speculative_rag`, add a `[tool.uv]` / `[tool.hatch.build.targets.wheel]` `packages = ["src/speculative_rag"]` entry (or `pip install -e .`-equivalent `uv` src layout config) to `pyproject.toml` matching the project's build backend, then re-run.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/speculative_rag/__init__.py src/speculative_rag/config.py tests/__init__.py
git commit -m "chore: scaffold speculative_rag package and settings"
```

---

### Task 2: Schemas

**Files:**
- Create: `src/speculative_rag/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing (no dependency on earlier tasks besides pydantic).
- Produces: `Passage(passage_id: str, text: str, language: str, query_type: str)`, `Chunk(chunk_id: str, text: str, language: str, query_type: str, source_passage_id: str, strategy: str)`, `QueryRecord(query_id: int, language: str, query_type: str, query_text: str, passage_ids: list[str], relevant_passage_ids: list[str])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas.py
from speculative_rag.schemas import Chunk, Passage, QueryRecord


def test_passage_roundtrip():
    p = Passage(passage_id="abc123", text="hello world", language="en", query_type="DESCRIPTION")
    assert p.passage_id == "abc123"
    assert p.model_dump()["text"] == "hello world"


def test_chunk_roundtrip():
    c = Chunk(
        chunk_id="abc123:native:0",
        text="hello world",
        language="en",
        query_type="DESCRIPTION",
        source_passage_id="abc123",
        strategy="native",
    )
    assert c.strategy == "native"


def test_query_record_roundtrip():
    q = QueryRecord(
        query_id=42,
        language="en",
        query_type="NUMERIC",
        query_text="how many?",
        passage_ids=["a", "b"],
        relevant_passage_ids=["b"],
    )
    assert q.relevant_passage_ids == ["b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.schemas'`

- [ ] **Step 3: Write `src/speculative_rag/schemas.py`**

```python
from pydantic import BaseModel


class Passage(BaseModel):
    passage_id: str
    text: str
    language: str
    query_type: str


class Chunk(BaseModel):
    chunk_id: str
    text: str
    language: str
    query_type: str
    source_passage_id: str
    strategy: str


class QueryRecord(BaseModel):
    query_id: int
    language: str
    query_type: str
    query_text: str
    passage_ids: list[str]
    relevant_passage_ids: list[str]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/schemas.py tests/test_schemas.py
git commit -m "feat: add Passage, Chunk, QueryRecord schemas"
```

---

### Task 3: Data loading, sampling, and passage dedup

**Files:**
- Create: `src/speculative_rag/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `Passage`, `QueryRecord` from `speculative_rag.schemas` (Task 2).
- Produces: `passage_id_for(language: str, text: str) -> str`, `load_validation_sample(parquet_path: Path, sample_size: int, seed: int) -> pandas.DataFrame`, `build_query_records(df, language: str, text_field: str) -> list[QueryRecord]`, `dedupe_passages(df, language: str, text_field: str) -> dict[str, Passage]`. Later tasks (5, 9) consume `dedupe_passages(...).values()` as the corpus to chunk, and `build_query_records(...)` as the eval queries.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
import numpy as np
import pandas as pd

from speculative_rag.data import (
    build_query_records,
    dedupe_passages,
    load_validation_sample,
    passage_id_for,
)


def _make_row(query_id, query_type, translated, english, is_selected):
    return {
        "query_id": query_id,
        "query_type": query_type,
        "query": f"query {query_id}",
        "passages": {
            "Translated_passages": np.array(translated, dtype=object),
            "English_passages": np.array(english, dtype=object),
            "is_selected": np.array(is_selected, dtype=np.int32),
        },
    }


def _make_df():
    rows = [
        _make_row(1, "DESCRIPTION", ["t1a", "t1b"], ["e1a", "e1b"], [0, 1]),
        _make_row(2, "NUMERIC", ["t2a"], ["e2a"], [0]),  # no gold passage
        _make_row(3, "ENTITY", ["t1a", "t3b"], ["e1a", "e3b"], [1, 0]),  # t1a repeats row 1
    ]
    return pd.DataFrame(rows)


def test_passage_id_is_deterministic_and_language_scoped():
    a = passage_id_for("hi", "same text")
    b = passage_id_for("hi", "same text")
    c = passage_id_for("en", "same text")
    assert a == b
    assert a != c


def test_load_validation_sample_filters_zero_selected_rows(tmp_path):
    df = _make_df()
    path = tmp_path / "sample.parquet"
    df.to_parquet(path)

    sampled = load_validation_sample(path, sample_size=10, seed=42)

    assert set(sampled["query_id"]) == {1, 3}


def test_build_query_records_maps_gold_passages():
    df = _make_df()
    records = build_query_records(df, language="hi", text_field="Translated_passages")

    by_id = {r.query_id: r for r in records}
    assert len(records) == 3
    assert by_id[1].relevant_passage_ids == [passage_id_for("hi", "t1b")]
    assert by_id[2].relevant_passage_ids == []
    assert by_id[3].relevant_passage_ids == [passage_id_for("hi", "t1a")]


def test_dedupe_passages_collapses_repeated_text():
    df = _make_df()
    passages = dedupe_passages(df, language="hi", text_field="Translated_passages")

    # t1a appears in rows 1 and 3 but must collapse to a single passage
    assert len(passages) == 3
    assert passage_id_for("hi", "t1a") in passages
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.data'`

- [ ] **Step 3: Write `src/speculative_rag/data.py`**

```python
import hashlib
from pathlib import Path

import pandas as pd

from speculative_rag.schemas import Passage, QueryRecord


def passage_id_for(language: str, text: str) -> str:
    digest = hashlib.sha1(f"{language}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def _has_gold_passage(passages: dict) -> bool:
    return sum(int(x) for x in passages["is_selected"]) > 0


def load_validation_sample(parquet_path: Path, sample_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df = df[df["passages"].apply(_has_gold_passage)]
    n = min(sample_size, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def build_query_records(df: pd.DataFrame, language: str, text_field: str) -> list[QueryRecord]:
    records = []
    for _, row in df.iterrows():
        passages = row["passages"]
        texts = list(passages[text_field])
        selected = [int(x) for x in passages["is_selected"]]
        passage_ids = [passage_id_for(language, t) for t in texts]
        relevant = [pid for pid, sel in zip(passage_ids, selected) if sel == 1]
        records.append(
            QueryRecord(
                query_id=int(row["query_id"]),
                language=language,
                query_type=row["query_type"],
                query_text=row["query"],
                passage_ids=passage_ids,
                relevant_passage_ids=relevant,
            )
        )
    return records


def dedupe_passages(df: pd.DataFrame, language: str, text_field: str) -> dict[str, Passage]:
    passages: dict[str, Passage] = {}
    for _, row in df.iterrows():
        for text in row["passages"][text_field]:
            pid = passage_id_for(language, text)
            if pid not in passages:
                passages[pid] = Passage(
                    passage_id=pid,
                    text=text,
                    language=language,
                    query_type=row["query_type"],
                )
    return passages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/data.py tests/test_data.py
git commit -m "feat: add parquet loading, sampling, and passage dedup"
```

---

### Task 4: Chunking strategies

**Files:**
- Create: `src/speculative_rag/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `Passage`, `Chunk` from `speculative_rag.schemas` (Task 2).
- Produces: `chunk_native(passage) -> list[Chunk]`, `chunk_sliding_window(passage, window_words=40, overlap_words=10) -> list[Chunk]`, `chunk_sentence_aware(passage, max_words=40, overlap_sentences=1) -> list[Chunk]`, `chunk_metadata_aware(passage) -> list[Chunk]`, `split_sentences(text: str) -> list[str]`, `CHUNKERS: dict[str, Callable[[Passage], list[Chunk]]]`. Task 9 iterates `CHUNKERS.items()` to build one collection per strategy.

**Design note (document in code as a module docstring, not just here):** these strategies chunk *within* one passage, never across passages in the same query's candidate set — MS MARCO's ~10 candidate passages per query come from unrelated source documents, so concatenating them would fabricate adjacency that isn't real. Most passages are short enough (spec: ~50-100 words) that `sliding_window`/`sentence_aware` degenerate to the same single chunk as `native` — that's an expected, honest empirical finding to report, not a bug. `metadata_aware` uses identical boundaries to `native`; its purpose is carrying explicit `language`/`query_type` tags for filtered retrieval, demonstrated qualitatively in Task 9's script, not a distinct chunk-boundary strategy.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking.py
from speculative_rag.chunking import (
    chunk_metadata_aware,
    chunk_native,
    chunk_sentence_aware,
    chunk_sliding_window,
    split_sentences,
)
from speculative_rag.schemas import Passage


def _passage(text):
    return Passage(passage_id="p1", text=text, language="en", query_type="DESCRIPTION")


def test_chunk_native_returns_single_chunk():
    chunks = chunk_native(_passage("A short passage."))
    assert len(chunks) == 1
    assert chunks[0].text == "A short passage."
    assert chunks[0].strategy == "native"
    assert chunks[0].source_passage_id == "p1"


def test_chunk_sliding_window_short_passage_equals_native():
    short = _passage(" ".join(["word"] * 10))
    chunks = chunk_sliding_window(short, window_words=40, overlap_words=10)
    assert len(chunks) == 1
    assert chunks[0].text == short.text


def test_chunk_sliding_window_long_passage_splits_with_overlap():
    long_text = " ".join(f"w{i}" for i in range(100))
    long_passage = _passage(long_text)
    chunks = chunk_sliding_window(long_passage, window_words=40, overlap_words=10)

    assert len(chunks) > 1
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-10:] == second_words[:10]  # overlap region matches
    assert all(c.source_passage_id == "p1" for c in chunks)


def test_split_sentences_handles_devanagari_danda():
    sentences = split_sentences("यह पहला वाक्य है। यह दूसरा वाक्य है।")
    assert len(sentences) == 2


def test_chunk_sentence_aware_short_passage_equals_native():
    short = _passage("One short sentence.")
    chunks = chunk_sentence_aware(short, max_words=40, overlap_sentences=1)
    assert len(chunks) == 1
    assert chunks[0].text == "One short sentence."


def test_chunk_sentence_aware_long_passage_splits_on_boundaries():
    sentences = [f"Sentence number {i} has several words in it." for i in range(10)]
    long_passage = _passage(" ".join(sentences))
    chunks = chunk_sentence_aware(long_passage, max_words=20, overlap_sentences=1)

    assert len(chunks) > 1
    for c in chunks:
        assert c.text.strip().endswith(".")


def test_chunk_metadata_aware_matches_native_boundaries_but_tags_strategy():
    p = _passage("A short passage.")
    native = chunk_native(p)[0]
    meta = chunk_metadata_aware(p)[0]

    assert meta.text == native.text
    assert meta.strategy == "metadata"
    assert meta.language == "en"
    assert meta.query_type == "DESCRIPTION"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.chunking'`

- [ ] **Step 3: Write `src/speculative_rag/chunking.py`**

```python
"""Chunking strategies operate within a single passage, never across passages
in the same query's candidate set — MS MARCO candidate passages come from
unrelated source documents, so concatenating them would fabricate adjacency
that isn't real. Most passages are short enough that sliding_window and
sentence_aware degenerate to the same single chunk as native; that is an
expected empirical finding, reported via Recall@K, not a bug.
"""

import re

from speculative_rag.schemas import Chunk, Passage

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text]


def chunk_native(passage: Passage) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{passage.passage_id}:native:0",
            text=passage.text,
            language=passage.language,
            query_type=passage.query_type,
            source_passage_id=passage.passage_id,
            strategy="native",
        )
    ]


def chunk_sliding_window(passage: Passage, window_words: int = 40, overlap_words: int = 10) -> list[Chunk]:
    words = passage.text.split()
    if len(words) <= window_words:
        return [
            Chunk(
                chunk_id=f"{passage.passage_id}:sliding_window:0",
                text=passage.text,
                language=passage.language,
                query_type=passage.query_type,
                source_passage_id=passage.passage_id,
                strategy="sliding_window",
            )
        ]

    step = window_words - overlap_words
    chunks = []
    idx = 0
    start = 0
    while start < len(words):
        window = words[start : start + window_words]
        chunks.append(
            Chunk(
                chunk_id=f"{passage.passage_id}:sliding_window:{idx}",
                text=" ".join(window),
                language=passage.language,
                query_type=passage.query_type,
                source_passage_id=passage.passage_id,
                strategy="sliding_window",
            )
        )
        idx += 1
        start += step
    return chunks


def chunk_sentence_aware(passage: Passage, max_words: int = 40, overlap_sentences: int = 1) -> list[Chunk]:
    sentences = split_sentences(passage.text)
    if len(sentences) <= 1:
        return [
            Chunk(
                chunk_id=f"{passage.passage_id}:sentence:0",
                text=passage.text,
                language=passage.language,
                query_type=passage.query_type,
                source_passage_id=passage.passage_id,
                strategy="sentence",
            )
        ]

    chunks = []
    idx = 0
    start = 0
    while start < len(sentences):
        group: list[str] = []
        word_count = 0
        i = start
        while i < len(sentences) and (word_count == 0 or word_count + len(sentences[i].split()) <= max_words):
            group.append(sentences[i])
            word_count += len(sentences[i].split())
            i += 1
        chunks.append(
            Chunk(
                chunk_id=f"{passage.passage_id}:sentence:{idx}",
                text=" ".join(group),
                language=passage.language,
                query_type=passage.query_type,
                source_passage_id=passage.passage_id,
                strategy="sentence",
            )
        )
        idx += 1
        if i >= len(sentences):
            break
        start = max(i - overlap_sentences, start + 1)
    return chunks


def chunk_metadata_aware(passage: Passage) -> list[Chunk]:
    native = chunk_native(passage)[0]
    return [native.model_copy(update={"chunk_id": f"{passage.passage_id}:metadata:0", "strategy": "metadata"})]


CHUNKERS = {
    "native": chunk_native,
    "sliding_window": chunk_sliding_window,
    "sentence": chunk_sentence_aware,
    "metadata": chunk_metadata_aware,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/chunking.py tests/test_chunking.py
git commit -m "feat: add native, sliding-window, sentence-aware, metadata chunkers"
```

---

### Task 5: Recall@K computation

**Files:**
- Create: `src/speculative_rag/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `Chunk` from `speculative_rag.schemas` (Task 2).
- Produces: `recall_hit(retrieved: list[Chunk], relevant_passage_ids: set[str], k: int) -> bool`, `compute_recall_at_k(hits: list[bool]) -> float`. Task 9 calls `recall_hit` once per query per k, collects the bools per `(strategy, language, k)`, then calls `compute_recall_at_k`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval.py
from speculative_rag.eval import compute_recall_at_k, recall_hit
from speculative_rag.schemas import Chunk


def _chunk(source_passage_id, strategy="native"):
    return Chunk(
        chunk_id=f"{source_passage_id}:{strategy}:0",
        text="text",
        language="en",
        query_type="DESCRIPTION",
        source_passage_id=source_passage_id,
        strategy=strategy,
    )


def test_recall_hit_true_when_relevant_passage_in_top_k():
    retrieved = [_chunk("a"), _chunk("b"), _chunk("c")]
    assert recall_hit(retrieved, {"c"}, k=3) is True


def test_recall_hit_false_when_relevant_passage_outside_top_k():
    retrieved = [_chunk("a"), _chunk("b"), _chunk("c")]
    assert recall_hit(retrieved, {"c"}, k=2) is False


def test_recall_hit_false_when_no_relevant_passages():
    retrieved = [_chunk("a")]
    assert recall_hit(retrieved, set(), k=1) is False


def test_compute_recall_at_k_averages_hits():
    assert compute_recall_at_k([True, True, False, False]) == 0.5


def test_compute_recall_at_k_empty_is_zero():
    assert compute_recall_at_k([]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.eval'`

- [ ] **Step 3: Write `src/speculative_rag/eval.py`**

```python
from speculative_rag.schemas import Chunk


def recall_hit(retrieved: list[Chunk], relevant_passage_ids: set[str], k: int) -> bool:
    top_k_source_ids = {c.source_passage_id for c in retrieved[:k]}
    return bool(top_k_source_ids & relevant_passage_ids)


def compute_recall_at_k(hits: list[bool]) -> float:
    if not hits:
        return 0.0
    return sum(hits) / len(hits)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/eval.py tests/test_eval.py
git commit -m "feat: add Recall@K computation"
```

---

### Task 6: Embedding wrapper

**Files:**
- Create: `src/speculative_rag/embedding.py`
- Test: `tests/test_embedding.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (plain strings in/out).
- Produces: `Embedder(model_name: str = SETTINGS.embedding_model)` with methods `embed_passages(texts: list[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`, and property `dimension: int`. Task 7 (`ChunkIndex`) consumes `dimension` for collection setup; Task 9 calls `embed_passages`/`embed_query`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedding.py
import pytest

from speculative_rag.embedding import Embedder


@pytest.mark.slow
def test_embed_query_and_passages_same_dimension():
    embedder = Embedder()
    query_vec = embedder.embed_query("what is a corporation?")
    passage_vecs = embedder.embed_passages(["a corporation is a company", "unrelated text about weather"])

    assert len(query_vec) == embedder.dimension
    assert all(len(v) == embedder.dimension for v in passage_vecs)


@pytest.mark.slow
def test_relevant_passage_scores_higher_than_irrelevant():
    import numpy as np

    embedder = Embedder()
    query_vec = np.array(embedder.embed_query("what is a corporation?"))
    relevant_vec = np.array(embedder.embed_passages(["A corporation is a company recognized in law."])[0])
    irrelevant_vec = np.array(embedder.embed_passages(["The weather today is sunny with light wind."])[0])

    assert float(query_vec @ relevant_vec) > float(query_vec @ irrelevant_vec)
```

- [ ] **Step 2: Register the `slow` marker and run test to verify it fails**

Add to `pyproject.toml` under a `[tool.pytest.ini_options]` table (create if absent):

```toml
[tool.pytest.ini_options]
markers = ["slow: integration tests that download models or hit local services"]
```

Run: `uv run pytest tests/test_embedding.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.embedding'`

- [ ] **Step 3: Write `src/speculative_rag/embedding.py`**

```python
from sentence_transformers import SentenceTransformer

from speculative_rag.config import SETTINGS


class Embedder:
    def __init__(self, model_name: str = SETTINGS.embedding_model):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embedding.py -v -m slow`
Expected: PASS (2 passed). First run downloads the ~470MB model — allow a few minutes.

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/embedding.py tests/test_embedding.py pyproject.toml uv.lock
git commit -m "feat: add multilingual-e5-small embedding wrapper"
```

---

### Task 7: Qdrant indexing

**Files:**
- Create: `src/speculative_rag/indexing.py`
- Test: `tests/test_indexing.py`

**Interfaces:**
- Consumes: `Chunk` from `speculative_rag.schemas` (Task 2).
- Produces: `ChunkIndex(client: QdrantClient, collection_name: str, vector_size: int)` with methods `add(chunks: list[Chunk], vectors: list[list[float]]) -> None` and `search(query_vector: list[float], k: int) -> list[Chunk]`. Task 9 creates one `QdrantClient(path=...)` shared across all strategy collections, and one `ChunkIndex` per strategy.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_indexing.py
import pytest
from qdrant_client import QdrantClient

from speculative_rag.indexing import ChunkIndex
from speculative_rag.schemas import Chunk


def _chunk(cid, text, source_passage_id):
    return Chunk(
        chunk_id=cid,
        text=text,
        language="en",
        query_type="DESCRIPTION",
        source_passage_id=source_passage_id,
        strategy="native",
    )


@pytest.mark.slow
def test_add_and_search_returns_nearest_chunk(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    index = ChunkIndex(client, collection_name="test", vector_size=3)

    chunks = [
        _chunk("c1", "near", "p1"),
        _chunk("c2", "far", "p2"),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    index.add(chunks, vectors)

    results = index.search(query_vector=[0.9, 0.1, 0.0], k=1)

    assert len(results) == 1
    assert results[0].source_passage_id == "p1"


@pytest.mark.slow
def test_search_respects_k(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    index = ChunkIndex(client, collection_name="test", vector_size=3)

    chunks = [_chunk(f"c{i}", f"text{i}", f"p{i}") for i in range(5)]
    vectors = [[float(i), 0.0, 0.0] for i in range(5)]
    index.add(chunks, vectors)

    results = index.search(query_vector=[2.0, 0.0, 0.0], k=3)

    assert len(results) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_indexing.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.indexing'`

- [ ] **Step 3: Write `src/speculative_rag/indexing.py`**

```python
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from speculative_rag.schemas import Chunk


class ChunkIndex:
    def __init__(self, client: QdrantClient, collection_name: str, vector_size: int):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_clean_collection()

    def _ensure_clean_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], k: int) -> list[Chunk]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
        ).points
        return [Chunk(**r.payload) for r in results]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_indexing.py -v -m slow`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/speculative_rag/indexing.py tests/test_indexing.py
git commit -m "feat: add local Qdrant ChunkIndex wrapper"
```

---

### Task 8: End-to-end comparison script

**Files:**
- Create: `scripts/run_recall_eval.py`
- Test: `tests/test_run_recall_eval.py` (tests the orchestration logic with a fake embedder/index, not real models)

**Interfaces:**
- Consumes: `LANGUAGE_SOURCES`, `SETTINGS` (Task 1); `load_validation_sample`, `build_query_records`, `dedupe_passages` (Task 3); `CHUNKERS` (Task 4); `recall_hit`, `compute_recall_at_k` (Task 5); `Embedder` (Task 6); `ChunkIndex` (Task 7).
- Produces: `run_comparison(language_sources, embedder, qdrant_client) -> list[dict]` — each dict has keys `language`, `strategy`, `k`, `recall`, `num_queries`, `num_chunks`. `main()` calls this with real components and writes `results/chunking_comparison.md`.

- [ ] **Step 1: Write the failing test using a fake embedder/index (no real model/network)**

```python
# tests/test_run_recall_eval.py
from qdrant_client import QdrantClient

from scripts.run_recall_eval import run_comparison
from speculative_rag.config import LanguageSource, Settings


class FakeEmbedder:
    """Deterministic hash-based embedding — no model download needed."""

    dimension = 8

    def _vec(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha1(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dimension]]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def test_run_comparison_produces_rows_for_every_strategy_and_k(tmp_path):
    import numpy as np
    import pandas as pd

    rows = [
        {
            "query_id": 1,
            "query_type": "DESCRIPTION",
            "query": "q1",
            "passages": {
                "Translated_passages": np.array(["gold passage text", "distractor text"], dtype=object),
                "English_passages": np.array(["gold passage text", "distractor text"], dtype=object),
                "is_selected": np.array([1, 0], dtype=np.int32),
            },
        }
    ]
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "fake.parquet"
    df.to_parquet(parquet_path)

    sources = [LanguageSource(language="hi", parquet_path=parquet_path, text_field="Translated_passages")]
    settings = Settings(sample_size=10, qdrant_path=tmp_path / "qdrant")
    client = QdrantClient(path=str(settings.qdrant_path))

    results = run_comparison(sources, FakeEmbedder(), client, settings)

    strategies = {r["strategy"] for r in results}
    assert strategies == {"native", "sliding_window", "sentence", "metadata"}
    assert all(r["num_queries"] == 1 for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run_recall_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_recall_eval'`

- [ ] **Step 3: Write `scripts/run_recall_eval.py`**

```python
from pathlib import Path

from qdrant_client import QdrantClient

from speculative_rag.chunking import CHUNKERS
from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS, LanguageSource, Settings
from speculative_rag.data import build_query_records, dedupe_passages, load_validation_sample
from speculative_rag.embedding import Embedder
from speculative_rag.eval import compute_recall_at_k, recall_hit
from speculative_rag.indexing import ChunkIndex


def run_comparison(
    language_sources: list[LanguageSource],
    embedder,
    qdrant_client: QdrantClient,
    settings: Settings,
) -> list[dict]:
    results: list[dict] = []

    for source in language_sources:
        df = load_validation_sample(source.parquet_path, settings.sample_size, settings.sample_seed)
        query_records = build_query_records(df, source.language, source.text_field)
        passages = list(dedupe_passages(df, source.language, source.text_field).values())

        for strategy_name, chunker in CHUNKERS.items():
            chunks = [c for passage in passages for c in chunker(passage)]
            vectors = embedder.embed_passages([c.text for c in chunks])

            collection_name = f"{source.language}_{strategy_name}"
            index = ChunkIndex(qdrant_client, collection_name, vector_size=embedder.dimension)
            index.add(chunks, vectors)

            for k in settings.top_k_values:
                hits = []
                for record in query_records:
                    if not record.relevant_passage_ids:
                        continue
                    query_vector = embedder.embed_query(record.query_text)
                    retrieved = index.search(query_vector, k=k)
                    hits.append(recall_hit(retrieved, set(record.relevant_passage_ids), k=k))

                results.append(
                    {
                        "language": source.language,
                        "strategy": strategy_name,
                        "k": k,
                        "recall": compute_recall_at_k(hits),
                        "num_queries": len(hits),
                        "num_chunks": len(chunks),
                    }
                )

    return results


def render_markdown_table(results: list[dict]) -> str:
    header = "| language | strategy | k | recall@k | num_queries | num_chunks |\n"
    header += "|---|---|---|---|---|---|\n"
    rows = "\n".join(
        f"| {r['language']} | {r['strategy']} | {r['k']} | {r['recall']:.3f} | {r['num_queries']} | {r['num_chunks']} |"
        for r in results
    )
    return header + rows + "\n"


def main() -> None:
    embedder = Embedder()
    qdrant_client = QdrantClient(path=str(SETTINGS.qdrant_path))

    results = run_comparison(LANGUAGE_SOURCES, embedder, qdrant_client, SETTINGS)

    SETTINGS.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = SETTINGS.results_dir / "chunking_comparison.md"
    output_path.write_text(render_markdown_table(results))
    print(render_markdown_table(results))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add `scripts/__init__.py` so it's importable in tests, then run test to verify it passes**

```bash
touch scripts/__init__.py
```

Run: `uv run pytest tests/test_run_recall_eval.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_recall_eval.py scripts/__init__.py tests/test_run_recall_eval.py
git commit -m "feat: add end-to-end chunking/recall comparison script"
```

---

### Task 9: Run the real evaluation and save results

**Files:**
- Create: `results/chunking_comparison.md` (generated, then committed)
- Modify: `.gitignore` (ignore `results/qdrant_local/`, keep `results/chunking_comparison.md`)

- [ ] **Step 1: Ignore the local Qdrant storage directory**

Add to `.gitignore`:

```
results/qdrant_local/
```

- [ ] **Step 2: Run the full evaluation against real data**

```bash
uv run python -m scripts.run_recall_eval
```

Expected: downloads the embedding model if not cached, processes ~300 queries × 3 languages × 4 strategies, prints a markdown table, and writes `results/chunking_comparison.md`. This will take several minutes (embedding + indexing several thousand chunks, then one query embedding per eval query).

- [ ] **Step 3: Sanity-check the output**

Read `results/chunking_comparison.md` and confirm: recall values are between 0 and 1 and non-degenerate (not all zero — a query embedded and searched against its own gold passage's chunk should retrieve it at `k=10` in the large majority of cases); `native`, `sliding_window`, `sentence`, `metadata` all have rows for `hi`, `en`, `ta`. If recall is near 0 across the board, stop and debug (systematic-debugging skill) rather than reporting a broken pipeline as "done" — likely causes: E5 prefix mismatch, passage_id hashing mismatch between dedup and query records, or Qdrant collection name collisions across strategies.

- [ ] **Step 4: Commit the results**

```bash
git add results/chunking_comparison.md .gitignore
git commit -m "docs: add Recall@K chunking strategy comparison results"
```

---

## Self-Review Notes

- **Spec coverage:** §1 (scoping to Hindi/English/Tamil, dedup) → Task 1 config + Task 3 dedup. §4 (four chunking strategies, Recall@K against `is_selected`) → Tasks 4, 5, 8, 9. §8 build-order steps 1-2 (data prep, indexing, Recall@K) → this entire plan. §4's hierarchical parent-child strategy is explicitly out of scope here (spec marks it "optional stretch") — not implemented, noted in Global Constraints.
- **Deliberate deviation from a literal reading of spec §4:** sliding-window/sentence-aware chunking operate within a passage, not across a query's candidate passages, because those candidates aren't contiguous source text. This is called out in Task 4's design note and Global Constraints so it isn't a silent gap against the spec.
- **Out of scope for this plan (separate future plans, per spec §2, §3, §5, §6, §7):** STT integration, hot-path retrieval+generation+guardrails harness, latency instrumentation/analytics. These depend on this phase's winning chunking strategy and embedding model choice, so they're deliberately sequenced after real Recall@K numbers exist.
