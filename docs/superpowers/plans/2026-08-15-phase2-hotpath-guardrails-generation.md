# Phase 2: Hot-Path RAG Pipeline — Retrieval, Guardrails & Extractive Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the timed hot-path RAG answering pipeline — text query in, extractive answer plus an explicit guardrail decision (answered / off-topic-refused / unsafe-input-refused / ungrounded-refused) and a per-stage latency breakdown out — reusing an existing Phase-1 Qdrant collection read-only, with no model calls in the timed path, and prove the <200ms target with a real P50/P70/P100 measurement over validation queries.

**Architecture:** A set of small, single-responsibility `src/speculative_rag/*.py` modules — pure-logic guardrails, span extraction, percentile math, and pipeline schemas that are unit-testable with zero network/model dependency — plus one thin read-only retrieval wrapper (`Retriever`, querying an existing collection via `QdrantClient(path=...).query_points(...)`, never recreating it) and one dependency-injected orchestrator (`run_pipeline`) that sequences the stages and captures per-stage timing. Two CLI scripts wire it together: an interactive answer REPL and a latency-analytics runner that writes a results table.

**Tech Stack:** Python 3.13, `uv` for dependency/venv management, `pydantic` for stage schemas, `sentence-transformers` (`intfloat/multilingual-e5-small`, reused from Phase 1) for query embedding, `qdrant-client` local on-disk mode (read-only query against Phase 1's index), `pytest` with the existing `slow` marker for integration tests. No new runtime dependencies.

## Global Constraints

- The **<200ms target covers the RAG pipeline only** — query embedding + vector retrieval + hot-path guardrail checks + extractive generation (spec §3). STT is out of scope this phase and is not timed here.
- **No model calls in the timed hot path** (spec §3, §6): every guardrail is heuristic (similarity threshold, lexical overlap, regex), and generation is extractive selection (rank/pick a passage span), never free-form generation or any hosted/local LLM call.
- **Both hot-path guardrail exits produce an explicit, typed refusal — never a silent fallback** (spec §6): off-topic → `RagOutcome.OFF_TOPIC_REFUSED`, unsafe-input → `RagOutcome.UNSAFE_INPUT_REFUSED`, each carrying a distinct `refusal_reason`. A third structural safety net (`UNGROUNDED_REFUSED`) guards the extraction step.
- **Target collection: `hi_native`** — the read-only default. Justification from `results/chunking_comparison.md`: Hindi `native` posts the best measured recall of any language×strategy (R@1=0.327, R@5=0.850, R@10=0.937); `native` and `metadata` are numerically identical (both one-chunk-per-passage), so `native` is chosen as the simpler control. One-chunk-per-passage means each retrieved chunk maps 1:1 to a source passage, which keeps extractive generation and the top-1 off-topic signal clean. The scripts accept `[language] [strategy]` args so `en_native`, `ta_native`, etc. are one flag away.
- **Retrieval is read-only**: query the existing collection directly with `QdrantClient(path=SETTINGS.qdrant_path).query_points(...)` (the pattern already in `scripts/query.py`). Do **not** use `ChunkIndex` — its constructor calls `_ensure_clean_collection()`, which destructively deletes and recreates the collection.
- Reuse Phase 1 as-is: `Embedder.embed_query(text) -> list[float]`, the `Chunk` schema, `chunking.split_sentences`, `data.load_validation_sample` / `data.build_query_records`, `config.SETTINGS` / `LANGUAGE_SOURCES`. Do not re-implement them.
- Use the **validation** split only for the latency eval (spec §7), sampled seeded, same as Phase 1.
- Use `uv add` / `uv run` for everything — this is a `uv`-managed project. No new runtime deps are required this phase.
- **Explicitly out of scope this phase** (deferred to Phase 3+): STT integration (spec §2, needs an unprovisioned external API key); the async LLM-based `ρ_Self-reflect` groundedness check (spec §6, non-hot-path); retries/exponential backoff and per-stage timeouts (spec §5 — no network-dependent leg exists in this hot path to attach them to); structured-logging infrastructure beyond what computes the latency numbers; the full three-way STT / RAG / async-check P50/P70/P100 report (spec §7 items 2-3 — this plan reports item 1, RAG-only, only); hierarchical/parent-child chunking.

---

## File Structure

- `src/speculative_rag/config.py` — **modify**: add Phase-2 fields to `Settings` (`off_topic_threshold`, `groundedness_threshold`, `default_language`, `default_strategy`, `answer_top_k`, `latency_sample_size`).
- `src/speculative_rag/text_utils.py` — **create**: `tokenize(text) -> set[str]`, a shared Unicode-aware token util (used by both guardrails and generation — DRY).
- `src/speculative_rag/pipeline_schemas.py` — **create**: `RagOutcome` enum, `ScoredChunk`, `RetrievalResult`, `Answer`, `StageTiming`, `PipelineResult` pydantic models.
- `src/speculative_rag/guardrails.py` — **create**: `check_unsafe_input`, `is_off_topic`, `groundedness_score` (all pure, no model calls).
- `src/speculative_rag/generation.py` — **create**: `select_answer_span` (extractive sentence-span selection, pure).
- `src/speculative_rag/retrieval.py` — **create**: `Retriever` (read-only query wrapper, does NOT create/delete collections).
- `src/speculative_rag/pipeline.py` — **create**: `run_pipeline` (DI orchestrator) + `_Timer` (per-stage timing).
- `src/speculative_rag/latency.py` — **create**: `percentile`, `summarize` (pure percentile math).
- `scripts/answer.py` — **create**: interactive extractive-answer REPL (parallel to `scripts/query.py`).
- `scripts/run_latency_eval.py` — **create**: RAG-only latency analytics runner; writes `results/latency_rag.md`.
- `tests/test_text_utils.py`, `tests/test_pipeline_schemas.py`, `tests/test_guardrails.py`, `tests/test_generation.py`, `tests/test_pipeline.py`, `tests/test_latency.py` — **create**: pure-logic unit tests (no network/model).
- `tests/test_retrieval.py` — **create**: `@pytest.mark.slow` integration test (temp local Qdrant, no model download).
- `results/latency_rag.md` — **create** (generated by Task 9, then committed).

---

### Task 1: Phase-2 config fields & shared token util

**Files:**
- Modify: `src/speculative_rag/config.py:14-24` (add fields to the `Settings` class body)
- Create: `src/speculative_rag/text_utils.py`
- Test: `tests/test_text_utils.py`

**Interfaces:**
- Consumes: existing `Settings(BaseModel)` from `config.py`.
- Produces: `Settings.off_topic_threshold: float`, `Settings.groundedness_threshold: float`, `Settings.default_language: str`, `Settings.default_strategy: str`, `Settings.answer_top_k: int`, `Settings.latency_sample_size: int`; `text_utils.tokenize(text: str) -> set[str]`.

- [ ] **Step 1: Write the failing test for `tokenize`**

Create `tests/test_text_utils.py`:
```python
from speculative_rag.text_utils import tokenize


def test_tokenize_lowercases_and_splits_on_word_boundaries():
    assert tokenize("A Corporation, is!") == {"a", "corporation", "is"}


def test_tokenize_empty_string_is_empty_set():
    assert tokenize("") == set()


def test_tokenize_handles_non_ascii_scripts():
    # \w with re.UNICODE keeps Devanagari word characters as tokens.
    tokens = tokenize("दिल्ली भारत")
    assert tokens == {"दिल्ली", "भारत"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_text_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.text_utils'`

- [ ] **Step 3: Write minimal implementation**

Create `src/speculative_rag/text_utils.py`:
```python
import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> set[str]:
    """Lowercased set of word tokens. Unicode-aware so it works for
    Devanagari/Tamil as well as Latin scripts (used by both the groundedness
    check and extractive span selection)."""
    return {match.lower() for match in _TOKEN_RE.findall(text)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_text_utils.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Add the Phase-2 `Settings` fields**

In `src/speculative_rag/config.py`, inside the `Settings` class, add these fields immediately after the existing `top_k_values: list[int] = [1, 5, 10]` line:
```python
    # Phase 2 (hot-path pipeline)
    off_topic_threshold: float = 0.75
    groundedness_threshold: float = 0.3
    default_language: str = "hi"
    default_strategy: str = "native"
    answer_top_k: int = 5
    latency_sample_size: int = 200
```

Notes for the implementer: `off_topic_threshold = 0.75` is a provisional default — Task 9's runner prints the real top-1 score distribution so it can be tuned to sit just below typical on-topic scores. `groundedness_threshold = 0.3` is a loose structural safety net: extractive answers are grounded ~1.0 by construction, so this only fires on a span-selection bug that returns text absent from the source. `default_language`/`default_strategy` encode the `hi_native` choice from Global Constraints.

- [ ] **Step 6: Verify config imports cleanly**

Run: `uv run python -c "from speculative_rag.config import SETTINGS; print(SETTINGS.off_topic_threshold, SETTINGS.default_language, SETTINGS.answer_top_k)"`
Expected: prints `0.75 hi 5`

- [ ] **Step 7: Commit**
```bash
git add src/speculative_rag/text_utils.py tests/test_text_utils.py src/speculative_rag/config.py
git commit -m "feat: add phase-2 config fields and shared tokenize util"
```

---

### Task 2: Pipeline stage schemas

**Files:**
- Create: `src/speculative_rag/pipeline_schemas.py`
- Test: `tests/test_pipeline_schemas.py`

**Interfaces:**
- Consumes: existing `Chunk(BaseModel)` from `speculative_rag.schemas`.
- Produces:
  - `RagOutcome(str, Enum)` with members `ANSWERED="answered"`, `OFF_TOPIC_REFUSED="off_topic_refused"`, `UNSAFE_INPUT_REFUSED="unsafe_input_refused"`, `UNGROUNDED_REFUSED="ungrounded_refused"`.
  - `ScoredChunk(BaseModel)`: `chunk: Chunk`, `score: float`.
  - `RetrievalResult(BaseModel)`: `scored_chunks: list[ScoredChunk]`; properties `top -> ScoredChunk | None`, `top_score -> float`.
  - `Answer(BaseModel)`: `text: str`, `source_passage_id: str`, `source_chunk_id: str`, `retrieval_score: float`, `groundedness_score: float`.
  - `StageTiming(BaseModel)`: `stage: str`, `elapsed_ms: float`.
  - `PipelineResult(BaseModel)`: `outcome: RagOutcome`, `answer: Answer | None = None`, `refusal_reason: str | None = None`, `top_score: float | None = None`, `timings: list[StageTiming]`; property `total_ms -> float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_schemas.py`:
```python
from speculative_rag.pipeline_schemas import (
    Answer,
    PipelineResult,
    RagOutcome,
    RetrievalResult,
    ScoredChunk,
    StageTiming,
)
from speculative_rag.schemas import Chunk


def _chunk(pid="p1"):
    return Chunk(
        chunk_id=f"{pid}:native:0",
        text="a corporation is a company",
        language="en",
        query_type="DESCRIPTION",
        source_passage_id=pid,
        strategy="native",
    )


def test_retrieval_result_top_and_top_score():
    result = RetrievalResult(
        scored_chunks=[
            ScoredChunk(chunk=_chunk("p1"), score=0.9),
            ScoredChunk(chunk=_chunk("p2"), score=0.4),
        ]
    )
    assert result.top.chunk.source_passage_id == "p1"
    assert result.top_score == 0.9


def test_retrieval_result_empty_top_is_none_and_score_zero():
    result = RetrievalResult(scored_chunks=[])
    assert result.top is None
    assert result.top_score == 0.0


def test_pipeline_result_total_ms_sums_timings():
    result = PipelineResult(
        outcome=RagOutcome.ANSWERED,
        answer=Answer(
            text="a corporation",
            source_passage_id="p1",
            source_chunk_id="p1:native:0",
            retrieval_score=0.9,
            groundedness_score=1.0,
        ),
        top_score=0.9,
        timings=[StageTiming(stage="embed", elapsed_ms=3.0), StageTiming(stage="retrieve", elapsed_ms=7.5)],
    )
    assert result.total_ms == 10.5
    assert result.outcome.value == "answered"


def test_pipeline_result_defaults_are_safe():
    result = PipelineResult(outcome=RagOutcome.UNSAFE_INPUT_REFUSED, refusal_reason="nope")
    assert result.answer is None
    assert result.timings == []
    assert result.total_ms == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.pipeline_schemas'`

- [ ] **Step 3: Write minimal implementation**

Create `src/speculative_rag/pipeline_schemas.py`:
```python
from enum import Enum

from pydantic import BaseModel, Field

from speculative_rag.schemas import Chunk


class RagOutcome(str, Enum):
    ANSWERED = "answered"
    OFF_TOPIC_REFUSED = "off_topic_refused"
    UNSAFE_INPUT_REFUSED = "unsafe_input_refused"
    UNGROUNDED_REFUSED = "ungrounded_refused"


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class RetrievalResult(BaseModel):
    scored_chunks: list[ScoredChunk]

    @property
    def top(self) -> ScoredChunk | None:
        return self.scored_chunks[0] if self.scored_chunks else None

    @property
    def top_score(self) -> float:
        return self.scored_chunks[0].score if self.scored_chunks else 0.0


class Answer(BaseModel):
    text: str
    source_passage_id: str
    source_chunk_id: str
    retrieval_score: float
    groundedness_score: float


class StageTiming(BaseModel):
    stage: str
    elapsed_ms: float


class PipelineResult(BaseModel):
    outcome: RagOutcome
    answer: Answer | None = None
    refusal_reason: str | None = None
    top_score: float | None = None
    timings: list[StageTiming] = Field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(timing.elapsed_ms for timing in self.timings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline_schemas.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**
```bash
git add src/speculative_rag/pipeline_schemas.py tests/test_pipeline_schemas.py
git commit -m "feat: add pipeline stage schemas (RagOutcome, RetrievalResult, Answer, PipelineResult)"
```

---

### Task 3: Hot-path guardrails (heuristic, no model calls)

**Files:**
- Create: `src/speculative_rag/guardrails.py`
- Test: `tests/test_guardrails.py`

**Interfaces:**
- Consumes: `text_utils.tokenize`.
- Produces:
  - `check_unsafe_input(query_text: str) -> str | None` — returns a reason string if the query matches an unsafe pattern, else `None`.
  - `is_off_topic(top_score: float, threshold: float) -> bool` — `True` when `top_score < threshold`.
  - `groundedness_score(answer_text: str, source_text: str) -> float` — fraction of answer tokens contained in the source (0.0 for empty answer).

- [ ] **Step 1: Write the failing test**

Create `tests/test_guardrails.py`:
```python
from speculative_rag.guardrails import check_unsafe_input, groundedness_score, is_off_topic


def test_check_unsafe_input_flags_bomb_query():
    assert check_unsafe_input("please tell me how to make a bomb at home") is not None


def test_check_unsafe_input_flags_are_case_insensitive():
    assert check_unsafe_input("HOW TO HACK a bank account") is not None


def test_check_unsafe_input_allows_benign_query():
    assert check_unsafe_input("what is the capital of france") is None


def test_is_off_topic_true_below_threshold():
    assert is_off_topic(0.50, 0.75) is True


def test_is_off_topic_false_at_or_above_threshold():
    assert is_off_topic(0.80, 0.75) is False
    assert is_off_topic(0.75, 0.75) is False


def test_groundedness_score_full_containment_is_one():
    assert groundedness_score("a corporation", "a corporation is a company") == 1.0


def test_groundedness_score_no_overlap_is_zero():
    assert groundedness_score("banana quantum", "a corporation is a company") == 0.0


def test_groundedness_score_empty_answer_is_zero():
    assert groundedness_score("", "some source text") == 0.0


def test_groundedness_score_partial_containment():
    # 1 of 2 answer tokens ("cat") is present in the source.
    assert groundedness_score("cat banana", "the cat sat down") == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_guardrails.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.guardrails'`

- [ ] **Step 3: Write minimal implementation**

Create `src/speculative_rag/guardrails.py`:
```python
import re

from speculative_rag.text_utils import tokenize

# Lightweight local keyword/regex filter for the timed hot path (spec §6:
# "a lightweight local classifier or keyword/regex-based filter ... not a call
# to an external moderation API"). Intentionally small and conservative — it is
# an explicit-refusal tripwire, not a full moderation model.
_UNSAFE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bhow to (make|build|create)\b.{0,30}\b(bomb|explosive|weapon|gun)\b", re.IGNORECASE),
    re.compile(r"\bhow to (hack|breach|break into)\b", re.IGNORECASE),
    re.compile(r"\b(kill|murder|poison)\b.{0,20}\b(someone|somebody|a person|people|him|her|them)\b", re.IGNORECASE),
    re.compile(r"\bchild\b.{0,15}\b(porn|sexual|abuse)\b", re.IGNORECASE),
    re.compile(r"\bhow to\b.{0,20}\bsuicide\b", re.IGNORECASE),
]


def check_unsafe_input(query_text: str) -> str | None:
    """Return a typed refusal reason if the query trips an unsafe pattern,
    else None. Runs before embedding so unsafe input never touches the index."""
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(query_text):
            return f"query matched unsafe-input pattern /{pattern.pattern}/"
    return None


def is_off_topic(top_score: float, threshold: float) -> bool:
    """Off-topic when the best retrieval score is below threshold — the corpus
    doesn't cover this query. Reuses the score already computed at retrieval."""
    return top_score < threshold


def groundedness_score(answer_text: str, source_text: str) -> float:
    """Fraction of the answer's tokens present in the source passage. For
    extractive answers this is ~1.0 by construction; a low value means the
    generated text drifted from its source. Lexical, no model call."""
    answer_tokens = tokenize(answer_text)
    if not answer_tokens:
        return 0.0
    source_tokens = tokenize(source_text)
    return len(answer_tokens & source_tokens) / len(answer_tokens)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_guardrails.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**
```bash
git add src/speculative_rag/guardrails.py tests/test_guardrails.py
git commit -m "feat: add heuristic hot-path guardrails (unsafe-input, off-topic, groundedness)"
```

---

### Task 4: Extractive span selection

**Files:**
- Create: `src/speculative_rag/generation.py`
- Test: `tests/test_generation.py`

**Interfaces:**
- Consumes: `chunking.split_sentences`, `text_utils.tokenize`.
- Produces: `select_answer_span(query_text: str, passage_text: str) -> str` — picks the sentence within the passage with the most token overlap with the query; returns the whole (stripped) passage when the passage is a single sentence or no sentence overlaps the query.

- [ ] **Step 1: Write the failing test**

Create `tests/test_generation.py`:
```python
from speculative_rag.generation import select_answer_span


def test_select_span_picks_sentence_overlapping_query():
    passage = "Cats are mammals. The Eiffel Tower is in Paris. Dogs bark loudly."
    span = select_answer_span("where is the Eiffel Tower", passage)
    assert "Eiffel Tower" in span
    assert "Cats" not in span


def test_select_span_single_sentence_returns_whole_passage():
    passage = "A corporation is a legal entity"
    assert select_answer_span("what is a corporation", passage) == "A corporation is a legal entity"


def test_select_span_no_overlap_falls_back_to_whole_passage():
    passage = "Cats are mammals. Dogs bark loudly."
    span = select_answer_span("quantum chromodynamics theory", passage)
    assert span == "Cats are mammals. Dogs bark loudly."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.generation'`

- [ ] **Step 3: Write minimal implementation**

Create `src/speculative_rag/generation.py`:
```python
"""Extractive answer selection (spec §3): rank and return the best-matching
span from the top retrieved passage — a ranking operation in single-digit
milliseconds, no model call, no free-form generation."""

from speculative_rag.chunking import split_sentences
from speculative_rag.text_utils import tokenize


def select_answer_span(query_text: str, passage_text: str) -> str:
    """Return the sentence in the passage with the most token overlap with the
    query. Falls back to the whole passage when it is a single sentence or no
    sentence shares any token with the query (span selection adds no signal)."""
    sentences = split_sentences(passage_text)
    if len(sentences) <= 1:
        return passage_text.strip()

    query_tokens = tokenize(query_text)

    def overlap(sentence: str) -> int:
        return len(tokenize(sentence) & query_tokens)

    best = max(sentences, key=overlap)
    if overlap(best) == 0:
        return passage_text.strip()
    return best.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generation.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**
```bash
git add src/speculative_rag/generation.py tests/test_generation.py
git commit -m "feat: add extractive span selection for answer generation"
```

---

### Task 5: Read-only retrieval wrapper

**Files:**
- Create: `src/speculative_rag/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `qdrant_client.QdrantClient`, `pipeline_schemas.RetrievalResult` / `ScoredChunk`, `schemas.Chunk`. (`ChunkIndex` from `indexing.py` is used **only in the test**, to seed a temp fixture collection.)
- Produces: `Retriever(client: QdrantClient, collection_name: str)` with `.search(query_vector: list[float], k: int) -> RetrievalResult`.

**Design note:** `Retriever` is deliberately read-only — it only calls `client.query_points(...)`, never `create_collection`/`delete_collection`. This is the whole reason it exists rather than reusing `ChunkIndex`, whose constructor destroys and recreates the collection. It mirrors the raw `query_points` pattern already in `scripts/query.py:16`, but wraps results in typed `ScoredChunk`s (preserving `score`, which the off-topic guardrail needs).

- [ ] **Step 1: Write the failing test**

Create `tests/test_retrieval.py`:
```python
import pytest
from qdrant_client import QdrantClient

from speculative_rag.indexing import ChunkIndex
from speculative_rag.retrieval import Retriever
from speculative_rag.schemas import Chunk


def _chunk(pid, text):
    return Chunk(
        chunk_id=f"{pid}:native:0",
        text=text,
        language="en",
        query_type="DESCRIPTION",
        source_passage_id=pid,
        strategy="native",
    )


@pytest.mark.slow
def test_retriever_returns_scored_chunks_ranked_by_similarity(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    # Seed a fixture collection with Phase-1's ChunkIndex (test-only helper).
    index = ChunkIndex(client, collection_name="en_native", vector_size=3)
    index.add(
        [_chunk("p1", "near"), _chunk("p2", "far")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    retriever = Retriever(client, "en_native")
    result = retriever.search([0.9, 0.1, 0.0], k=2)

    assert len(result.scored_chunks) == 2
    assert result.top.chunk.source_passage_id == "p1"
    assert result.top_score > result.scored_chunks[1].score


@pytest.mark.slow
def test_retriever_respects_k(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    index = ChunkIndex(client, collection_name="en_native", vector_size=3)
    index.add(
        [_chunk(f"p{i}", f"text{i}") for i in range(5)],
        [[float(i), 0.0, 0.0] for i in range(5)],
    )

    retriever = Retriever(client, "en_native")
    result = retriever.search([2.0, 0.0, 0.0], k=3)

    assert len(result.scored_chunks) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.retrieval'`

- [ ] **Step 3: Write minimal implementation**

Create `src/speculative_rag/retrieval.py`:
```python
"""Read-only retrieval against an EXISTING Phase-1 collection. Unlike
indexing.ChunkIndex (whose constructor deletes and recreates the collection),
Retriever only queries — it never mutates the index."""

from qdrant_client import QdrantClient

from speculative_rag.pipeline_schemas import RetrievalResult, ScoredChunk
from speculative_rag.schemas import Chunk


class Retriever:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def search(self, query_vector: list[float], k: int) -> RetrievalResult:
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
        ).points
        scored = [ScoredChunk(chunk=Chunk(**point.payload), score=point.score) for point in points]
        return RetrievalResult(scored_chunks=scored)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**
```bash
git add src/speculative_rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add read-only Retriever over an existing Qdrant collection"
```

---

### Task 6: Pipeline orchestrator with per-stage timing

**Files:**
- Create: `src/speculative_rag/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `guardrails.check_unsafe_input` / `is_off_topic` / `groundedness_score`, `generation.select_answer_span`, `pipeline_schemas.*`. Depends on two injected collaborators (duck-typed for testability): an `embedder` with `.embed_query(text: str) -> list[float]` (satisfied by Phase-1 `Embedder`) and a `retriever` with `.search(query_vector: list[float], k: int) -> RetrievalResult` (satisfied by Task-5 `Retriever`).
- Produces: `run_pipeline(query_text: str, embedder, retriever, *, off_topic_threshold: float, groundedness_threshold: float, top_k: int) -> PipelineResult`.

**Stage order & exits:** `guardrail_unsafe` (before embedding — cheapest, and unsafe input must never touch the index) → `embed` → `retrieve` → `guardrail_offtopic` (reuses the retrieval top score) → `generate` → `guardrail_groundedness`. Refusals short-circuit and return immediately with the stages timed so far, so `PipelineResult.timings` always reflects exactly the work actually done.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:
```python
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome, RetrievalResult, ScoredChunk
from speculative_rag.schemas import Chunk


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0, 0.0, 0.0]


class FakeRetriever:
    def __init__(self, result: RetrievalResult):
        self.result = result

    def search(self, query_vector, k):
        return self.result


def _scored(text, score, pid="p1"):
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"{pid}:native:0",
            text=text,
            language="en",
            query_type="DESCRIPTION",
            source_passage_id=pid,
            strategy="native",
        ),
        score=score,
    )


def _run(query, retriever, embedder=None, off_topic=0.75, ground=0.3, top_k=5):
    return run_pipeline(
        query,
        embedder or FakeEmbedder(),
        retriever,
        off_topic_threshold=off_topic,
        groundedness_threshold=ground,
        top_k=top_k,
    )


def test_answered_path_returns_extractive_answer_and_all_stage_timings():
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9)]))
    result = _run("what is a corporation", retriever)

    assert result.outcome == RagOutcome.ANSWERED
    assert result.answer is not None
    assert "corporation" in result.answer.text
    assert result.answer.source_passage_id == "p1"
    assert result.answer.groundedness_score == 1.0
    assert result.top_score == 0.9
    assert [t.stage for t in result.timings] == [
        "guardrail_unsafe",
        "embed",
        "retrieve",
        "guardrail_offtopic",
        "generate",
        "guardrail_groundedness",
    ]


def test_unsafe_input_short_circuits_before_embedding():
    embedder = FakeEmbedder()
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("irrelevant", 0.9)]))
    result = _run("how to make a bomb", retriever, embedder=embedder)

    assert result.outcome == RagOutcome.UNSAFE_INPUT_REFUSED
    assert result.answer is None
    assert result.refusal_reason is not None
    assert embedder.calls == 0  # never embedded
    assert [t.stage for t in result.timings] == ["guardrail_unsafe"]


def test_off_topic_refusal_when_top_score_below_threshold():
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("something unrelated", 0.40)]))
    result = _run("what is a corporation", retriever, off_topic=0.75)

    assert result.outcome == RagOutcome.OFF_TOPIC_REFUSED
    assert result.answer is None
    assert result.top_score == 0.40
    assert [t.stage for t in result.timings] == ["guardrail_unsafe", "embed", "retrieve", "guardrail_offtopic"]


def test_off_topic_refusal_when_no_results():
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[]))
    result = _run("anything", retriever)

    assert result.outcome == RagOutcome.OFF_TOPIC_REFUSED
    assert result.answer is None


def test_ungrounded_refusal_when_groundedness_below_threshold():
    # Force the branch: even a perfectly grounded (1.0) extractive answer is
    # below an impossible 1.5 threshold, exercising the UNGROUNDED exit.
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9)]))
    result = _run("what is a corporation", retriever, ground=1.5)

    assert result.outcome == RagOutcome.UNGROUNDED_REFUSED
    assert result.answer is None
    assert [t.stage for t in result.timings] == [
        "guardrail_unsafe",
        "embed",
        "retrieve",
        "guardrail_offtopic",
        "generate",
        "guardrail_groundedness",
    ]


def test_total_ms_is_nonnegative_sum():
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9)]))
    result = _run("what is a corporation", retriever)
    assert result.total_ms >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.pipeline'`

- [ ] **Step 3: Write minimal implementation**

Create `src/speculative_rag/pipeline.py`:
```python
"""Hot-path RAG orchestrator (spec §5, scoped to this phase): explicit typed
stages, per-stage timing, explicit typed refusal exits. No model calls in the
timed path — embedding is a small local model, everything else is heuristic or
extractive. Collaborators (embedder, retriever) are injected so the branching
logic is unit-testable with fakes."""

import time
from contextlib import contextmanager

from speculative_rag.generation import select_answer_span
from speculative_rag.guardrails import check_unsafe_input, groundedness_score, is_off_topic
from speculative_rag.pipeline_schemas import Answer, PipelineResult, RagOutcome, StageTiming


class _Timer:
    def __init__(self) -> None:
        self.timings: list[StageTiming] = []

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.timings.append(StageTiming(stage=name, elapsed_ms=elapsed_ms))


def run_pipeline(
    query_text: str,
    embedder,
    retriever,
    *,
    off_topic_threshold: float,
    groundedness_threshold: float,
    top_k: int,
) -> PipelineResult:
    timer = _Timer()

    with timer.stage("guardrail_unsafe"):
        unsafe_reason = check_unsafe_input(query_text)
    if unsafe_reason is not None:
        return PipelineResult(
            outcome=RagOutcome.UNSAFE_INPUT_REFUSED,
            refusal_reason=unsafe_reason,
            timings=timer.timings,
        )

    with timer.stage("embed"):
        query_vector = embedder.embed_query(query_text)

    with timer.stage("retrieve"):
        retrieval = retriever.search(query_vector, k=top_k)

    top = retrieval.top
    top_score = retrieval.top_score

    with timer.stage("guardrail_offtopic"):
        off_topic = top is None or is_off_topic(top_score, off_topic_threshold)
    if off_topic:
        return PipelineResult(
            outcome=RagOutcome.OFF_TOPIC_REFUSED,
            refusal_reason=(
                f"top retrieval score {top_score:.3f} below off-topic threshold {off_topic_threshold:.3f}"
            ),
            top_score=top_score,
            timings=timer.timings,
        )

    with timer.stage("generate"):
        span = select_answer_span(query_text, top.chunk.text)

    with timer.stage("guardrail_groundedness"):
        ground = groundedness_score(span, top.chunk.text)
    if ground < groundedness_threshold:
        return PipelineResult(
            outcome=RagOutcome.UNGROUNDED_REFUSED,
            refusal_reason=(
                f"groundedness {ground:.3f} below threshold {groundedness_threshold:.3f}"
            ),
            top_score=top_score,
            timings=timer.timings,
        )

    answer = Answer(
        text=span,
        source_passage_id=top.chunk.source_passage_id,
        source_chunk_id=top.chunk.chunk_id,
        retrieval_score=top_score,
        groundedness_score=ground,
    )
    return PipelineResult(
        outcome=RagOutcome.ANSWERED,
        answer=answer,
        top_score=top_score,
        timings=timer.timings,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**
```bash
git add src/speculative_rag/pipeline.py tests/test_pipeline.py
git commit -m "feat: add hot-path pipeline orchestrator with per-stage timing and typed refusals"
```

---

### Task 7: Percentile / latency summary math

**Files:**
- Create: `src/speculative_rag/latency.py`
- Test: `tests/test_latency.py`

**Interfaces:**
- Produces:
  - `percentile(values: list[float], pct: float) -> float` — linear-interpolated percentile; `pct >= 100` returns the max; empty list returns `0.0`.
  - `summarize(values: list[float]) -> dict[str, float]` — `{"p50": ..., "p70": ..., "p100": ...}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_latency.py`:
```python
from speculative_rag.latency import percentile, summarize


def test_percentile_median_of_odd_list():
    assert percentile([1, 2, 3, 4, 5], 50) == 3


def test_percentile_p100_is_max():
    assert percentile([1, 5, 2, 9, 3], 100) == 9


def test_percentile_empty_is_zero():
    assert percentile([], 50) == 0.0


def test_percentile_interpolates():
    # 50th pct of 1..10: rank = 0.5*(10-1) = 4.5 -> 5 + 0.5*(6-5) = 5.5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5.5


def test_summarize_returns_ordered_p50_p70_p100():
    summary = summarize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert summary["p100"] == 10
    assert summary["p50"] <= summary["p70"] <= summary["p100"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_latency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speculative_rag.latency'`

- [ ] **Step 3: Write minimal implementation**

Create `src/speculative_rag/latency.py`:
```python
"""Percentile math for the RAG-only latency report (spec §7, item 1).
P100 is the single worst observed value — reported as required, but P50/P70
describe typical behaviour."""


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if pct >= 100:
        return ordered[-1]
    if pct <= 0:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 50),
        "p70": percentile(values, 70),
        "p100": percentile(values, 100),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_latency.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**
```bash
git add src/speculative_rag/latency.py tests/test_latency.py
git commit -m "feat: add percentile/summarize math for RAG latency report"
```

---

### Task 8: Interactive extractive-answer CLI

**Files:**
- Create: `scripts/answer.py`

**Interfaces:**
- Consumes: `SETTINGS`, `Embedder`, `QdrantClient`, `Retriever`, `run_pipeline`, `RagOutcome`.
- Produces: a runnable module `scripts.answer` (no importable API beyond `main()`). Parallel to `scripts/query.py`.

**Note:** There is no unit test for this task — it's a thin I/O wrapper (same convention as Phase 1's `scripts/query.py`, which has no test). It is exercised manually in Step 2 and end-to-end by Task 9's runner. Keep all logic in the already-tested library modules.

- [ ] **Step 1: Write `scripts/answer.py`**

Create `scripts/answer.py`:
```python
"""Interactive extractive-answer CLI over an EXISTING Phase-1 collection.
Text query in -> extractive answer + guardrail decision + per-stage latency.
Read-only: never rebuilds the collection. Usage:
    uv run python -m scripts.answer [language] [strategy]
Defaults to the SETTINGS.default_language / default_strategy collection (hi_native).
"""

import sys

from qdrant_client import QdrantClient

from speculative_rag.config import SETTINGS
from speculative_rag.embedding import Embedder
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.retrieval import Retriever


def main() -> None:
    language = sys.argv[1] if len(sys.argv) > 1 else SETTINGS.default_language
    strategy = sys.argv[2] if len(sys.argv) > 2 else SETTINGS.default_strategy
    collection_name = f"{language}_{strategy}"

    print(f"Loading embedder and connecting to collection '{collection_name}'...")
    embedder = Embedder()
    client = QdrantClient(path=str(SETTINGS.qdrant_path))

    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        print(f"No such collection '{collection_name}'. Available: {available}")
        print("Run `uv run python -m scripts.run_recall_eval` first to build the index.")
        return

    retriever = Retriever(client, collection_name)

    print("Type a question (in the collection's language), Ctrl+D to quit.\n")
    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            print()
            break
        if not query:
            continue

        result = run_pipeline(
            query,
            embedder,
            retriever,
            off_topic_threshold=SETTINGS.off_topic_threshold,
            groundedness_threshold=SETTINGS.groundedness_threshold,
            top_k=SETTINGS.answer_top_k,
        )

        print(f"\noutcome: {result.outcome.value}")
        if result.outcome == RagOutcome.ANSWERED:
            print(f"answer: {result.answer.text}")
            print(
                f"source passage: {result.answer.source_passage_id}  "
                f"score={result.answer.retrieval_score:.3f}  "
                f"groundedness={result.answer.groundedness_score:.3f}"
            )
        else:
            print(f"refused: {result.refusal_reason}")

        print("latency (ms):")
        for timing in result.timings:
            print(f"  {timing.stage:24s} {timing.elapsed_ms:7.2f}")
        print(f"  {'TOTAL':24s} {result.total_ms:7.2f}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI against the real `hi_native` collection**

Run (pipes two queries in, then EOF):
```bash
printf 'भारत की राजधानी क्या है\nhow to make a bomb\n' | uv run python -m scripts.answer hi native
```
Expected: for the first (Hindi) query, an `outcome:` line (`answered` or `off_topic_refused`) plus a latency table whose `TOTAL` prints; for the second, `outcome: unsafe_input_refused` with a `refused:` line and only a `guardrail_unsafe` timing. If it prints `No such collection 'hi_native'`, the Phase-1 index isn't present — run `uv run python -m scripts.run_recall_eval` first.

- [ ] **Step 3: Commit**
```bash
git add scripts/answer.py
git commit -m "feat: add interactive extractive-answer CLI (scripts.answer)"
```

---

### Task 9: RAG-only latency analytics runner + real measurement

**Files:**
- Create: `scripts/run_latency_eval.py`
- Create: `results/latency_rag.md` (generated by Step 3, then committed)
- Modify: `.gitignore` if needed (the existing `results/qdrant_local/` ignore from Phase 1 must stay; `results/latency_rag.md` must be committable — verify it isn't ignored)

**Interfaces:**
- Consumes: `LANGUAGE_SOURCES`, `SETTINGS`, `data.load_validation_sample`, `data.build_query_records`, `Embedder`, `Retriever`, `run_pipeline`, `RagOutcome`, `latency.summarize`.
- Produces: a runnable module `scripts.run_latency_eval` and the report `results/latency_rag.md`.

**Note:** Like Phase 1's `scripts/run_recall_eval.py`, the thin `main()` glue isn't unit-tested; the logic it calls (`summarize`, `run_pipeline`, guardrails, generation) is already covered. Its correctness is validated by the sanity check in Step 4.

- [ ] **Step 1: Write `scripts/run_latency_eval.py`**

Create `scripts/run_latency_eval.py`:
```python
"""RAG-pipeline-only latency analytics (spec §7, item 1 only — NOT the STT or
async-check numbers, which are later phases).

Samples validation queries in the default collection's language, runs the
hot-path pipeline per query against the EXISTING Phase-1 collection
(read-only), and reports P50/P70/P100 of total RAG-pipeline latency plus a
guardrail-outcome breakdown and the top-1 score distribution (to justify the
off-topic threshold). Proves whether the <200ms target holds.

Usage: uv run python -m scripts.run_latency_eval
"""

from qdrant_client import QdrantClient

from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS
from speculative_rag.data import build_query_records, load_validation_sample
from speculative_rag.embedding import Embedder
from speculative_rag.latency import summarize
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.retrieval import Retriever


def _language_source(language: str):
    return next(source for source in LANGUAGE_SOURCES if source.language == language)


def _run(text: str, embedder, retriever):
    return run_pipeline(
        text,
        embedder,
        retriever,
        off_topic_threshold=SETTINGS.off_topic_threshold,
        groundedness_threshold=SETTINGS.groundedness_threshold,
        top_k=SETTINGS.answer_top_k,
    )


def render_report(collection_name: str, latencies: list[float], outcomes: dict[str, int], top_scores: list[float]) -> str:
    stats = summarize(latencies)
    lines = [
        f"# RAG-pipeline-only latency — `{collection_name}`, {len(latencies)} validation queries",
        "",
        "| metric | value |",
        "|---|---|",
        f"| P50 (ms) | {stats['p50']:.2f} |",
        f"| P70 (ms) | {stats['p70']:.2f} |",
        f"| P100 (ms) | {stats['p100']:.2f} |",
        "| target | <200ms |",
        f"| P100 within target | {'YES' if stats['p100'] < 200 else 'NO'} |",
        "",
        "## Guardrail outcome breakdown",
        "",
        "| outcome | count |",
        "|---|---|",
        *[f"| {name} | {count} |" for name, count in outcomes.items()],
        "",
    ]
    if top_scores:
        ordered = sorted(top_scores)
        median = ordered[len(ordered) // 2]
        lines.append(
            f"top-1 retrieval score across scored queries: min={ordered[0]:.3f} "
            f"median={median:.3f} max={ordered[-1]:.3f} "
            f"(off_topic_threshold={SETTINGS.off_topic_threshold})"
        )
    else:
        lines.append("no queries reached the retrieval stage (all refused pre-embedding)")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    language = SETTINGS.default_language
    strategy = SETTINGS.default_strategy
    collection_name = f"{language}_{strategy}"

    source = _language_source(language)
    df = load_validation_sample(source.parquet_path, SETTINGS.latency_sample_size, SETTINGS.sample_seed)
    records = build_query_records(df, source.language, source.text_field)

    embedder = Embedder()
    client = QdrantClient(path=str(SETTINGS.qdrant_path))
    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        print(f"No such collection '{collection_name}'. Available: {available}")
        print("Run `uv run python -m scripts.run_recall_eval` first to build the index.")
        return
    retriever = Retriever(client, collection_name)

    # Warm the embedder and the on-disk index so the first timed query isn't
    # penalised by lazy model / segment loading (spec §3: keep the index warm).
    _run(records[0].query_text, embedder, retriever)

    latencies: list[float] = []
    top_scores: list[float] = []
    outcomes: dict[str, int] = {outcome.value: 0 for outcome in RagOutcome}
    for record in records:
        result = _run(record.query_text, embedder, retriever)
        latencies.append(result.total_ms)
        outcomes[result.outcome.value] += 1
        if result.top_score is not None:
            top_scores.append(result.top_score)

    report = render_report(collection_name, latencies, outcomes, top_scores)
    SETTINGS.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = SETTINGS.results_dir / "latency_rag.md"
    output_path.write_text(report)
    print(report)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the Phase-1 index is present**

Run: `uv run python -c "from qdrant_client import QdrantClient; from speculative_rag.config import SETTINGS; c = QdrantClient(path=str(SETTINGS.qdrant_path)); print(sorted(col.name for col in c.get_collections().collections))"`
Expected: a list including `hi_native` (and the other 11 `{lang}_{strategy}` collections). If `hi_native` is absent, run `uv run python -m scripts.run_recall_eval` first to rebuild the Phase-1 index.

- [ ] **Step 3: Run the real latency measurement**

Run: `uv run python -m scripts.run_latency_eval`
Expected: downloads the embedding model if not cached, runs ~200 validation queries through the hot path against `hi_native`, prints the P50/P70/P100 table + outcome breakdown + top-score line, and writes `results/latency_rag.md`.

- [ ] **Step 4: Sanity-check the output**

Read `results/latency_rag.md` and confirm:
- P50/P70/P100 are positive and P100 is comfortably under 200ms (an all-local embed + on-disk vector search + heuristic checks + extraction should land in the low tens of ms; if P100 exceeds 200ms, do NOT report success — investigate with systematic-debugging: likely a cold index/model not warmed, or an accidental model call in a stage).
- The `answered` count dominates the outcome breakdown (validation queries are real, on-topic queries whose gold passage is in the corpus). If nearly everything is `off_topic_refused`, the `off_topic_threshold` (0.75) is set too high for the measured score distribution printed on the last line — lower `Settings.off_topic_threshold` in `config.py` to sit just below the reported median, re-run, and note the tuned value in the commit message.
- `unsafe_input_refused` and `ungrounded_refused` are expected to be 0 or near-0 on this benign validation sample — that's correct, not a bug (their behaviour is proven by the Task 3/Task 6 unit tests, not by this benign corpus).

- [ ] **Step 5: Verify the report is committable (not gitignored)**

Run: `git check-ignore results/latency_rag.md; echo "exit=$?"`
Expected: prints `exit=1` (i.e. NOT ignored). If it prints the path with `exit=0`, the Phase-1 `.gitignore` rule is too broad — narrow it so only `results/qdrant_local/` is ignored, keeping `results/*.md` tracked.

- [ ] **Step 6: Commit the runner and results**
```bash
git add scripts/run_latency_eval.py results/latency_rag.md
git commit -m "feat: add RAG-only latency runner and committed P50/P70/P100 results"
```

---

### Task 10: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the fast unit suite (no model/network)**

Run: `uv run pytest -m "not slow" -v`
Expected: all pure-logic tests pass (`test_text_utils`, `test_pipeline_schemas`, `test_guardrails`, `test_generation`, `test_pipeline`, `test_latency`, plus the Phase-1 pure tests). Zero failures.

- [ ] **Step 2: Run the full suite including slow integration tests**

Run: `uv run pytest -v`
Expected: all tests pass, including `test_retrieval` (temp Qdrant) and the Phase-1 slow tests. Zero failures.

- [ ] **Step 3: Confirm no accidental destructive collection access**

Run: `git grep -n "ChunkIndex" src/speculative_rag/pipeline.py src/speculative_rag/retrieval.py scripts/answer.py scripts/run_latency_eval.py`
Expected: no matches (the hot path and its scripts never touch the destructive `ChunkIndex` — only `tests/test_retrieval.py` uses it, to seed a temp fixture).

- [ ] **Step 4: Commit (only if any fixups were needed above)**
```bash
git add -A
git commit -m "test: verify full phase-2 suite passes and hot path is non-destructive"
```

---

## Self-Review Notes

**Spec coverage walkthrough:**
- **§3 (latency budget composition):** the timed stages are exactly `embed` + `retrieve` + hot-path guardrails (`guardrail_unsafe`, `guardrail_offtopic`, `guardrail_groundedness`) + `generate` → Task 6 (`run_pipeline`) captures each; Task 9 measures P50/P70/P100 and asserts <200ms. "No hosted/local LLM in the hot path" → generation is extractive (Task 4), all guardrails heuristic (Task 3). "Keep the index warm" → Task 9 warm-up call. "Extractive/near-extractive as primary path" → Task 4.
- **§5 (explicit typed stages, scoped down):** each stage is its own function with a Pydantic input/output contract (`RetrievalResult`, `Answer`, `PipelineResult`, `StageTiming`) → Tasks 2 + 6. Per-stage timing → `_Timer` in Task 6. **Deliberately deferred (stated in Global Constraints):** retries/backoff and per-stage timeouts — there is no network-dependent leg in this hot path (embedding + on-disk Qdrant are local) for them to attach to; structured-logging infra beyond the latency numbers.
- **§6 (hot-path guardrails, all three, each with explicit typed refusal):** off-topic reuses the retrieval top score (Task 3 `is_off_topic`, Task 6 `guardrail_offtopic` → `OFF_TOPIC_REFUSED`); groundedness is lexical answer-vs-source overlap (Task 3 `groundedness_score`, Task 6 → `UNGROUNDED_REFUSED`); unsafe-input is a local regex filter (Task 3 `check_unsafe_input`, Task 6 → `UNSAFE_INPUT_REFUSED`). All three are heuristic, no model calls. Each has a distinct `RagOutcome` and a populated `refusal_reason` — explicit refusal, never silent fallback. **Deferred (Global Constraints):** the async LLM-based `ρ_Self-reflect` groundedness check (spec §6 non-hot-path).
- **§7 (latency analytics):** item 1 only — RAG-pipeline P50/P70/P100 over a seeded 200-query validation sample → Task 9. **Deferred:** items 2-3 (STT and async-check numbers, the three-way report).

**On the groundedness check being ~1.0 by construction:** for a purely extractive answer, `groundedness_score(span, passage) ≈ 1.0` because the span is a substring of the passage. This is honest and intentional — it's a structural safety net (`groundedness_threshold = 0.3`) that only fires if span selection ever returns text not present in the source (e.g. a future refactor introducing paraphrase). It's implemented as a real typed check with a real threshold and a real refusal path, satisfying §6's "explicit refusal" requirement, and its branch is proven by `test_ungrounded_refusal_...` (which forces the threshold above 1.0). This is called out here so it isn't mistaken for a silently dead check.

**On the `hi_native` default:** chosen strictly from `results/chunking_comparison.md` — Hindi `native` has the best measured recall of any language×strategy (R@5=0.850, R@10=0.937); `metadata` is numerically identical (same one-chunk-per-passage output) so `native` is picked as the simpler control. Tradeoff acknowledged: a developer typing English queries would prefer `en_native` (cross-lingual E5 retrieval into a Hindi collection degrades), so both scripts accept `[language] [strategy]` args and `en_native`/`ta_native` are a single flag away. The default follows the measured numbers as instructed.

**Placeholder scan:** no "TBD"/"implement later"/"add appropriate error handling"/"similar to Task N" — every step contains complete, runnable code and every referenced type (`RagOutcome`, `ScoredChunk`, `RetrievalResult`, `Answer`, `StageTiming`, `PipelineResult`, `Retriever`, `run_pipeline`, `tokenize`, `percentile`, `summarize`) is defined in an earlier task.

**Type-consistency check:** `RagOutcome` string values (`answered`, `off_topic_refused`, `unsafe_input_refused`, `ungrounded_refused`) are used identically in Tasks 2, 6, 8, 9. `RetrievalResult.top` / `top_score` (properties) are consumed the same way in `run_pipeline` and the CLI. `PipelineResult.timings` / `total_ms` are produced by `_Timer` and consumed identically by both scripts. `Answer` field names (`text`, `source_passage_id`, `source_chunk_id`, `retrieval_score`, `groundedness_score`) match between the schema (Task 2), its construction in `run_pipeline` (Task 6), and the CLI print (Task 8). The injected `embedder`/`retriever` duck-typed contracts (`embed_query`, `search`) match Phase-1 `Embedder` and Task-5 `Retriever` exactly.

**Out of scope for this plan (future phases, per spec §2, §5, §6, §7):** STT integration (needs an unprovisioned external key); async LLM `ρ_Self-reflect` groundedness; retries/timeouts/structured-logging harness; the three-way STT/RAG/async P50/P70/P100 report; hierarchical/parent-child chunking. These are stated in Global Constraints and depend on artifacts (an STT key, a hosted verification model) not available this phase.

---

Plan complete and saved to `docs/superpowers/plans/2026-08-15-phase2-hotpath-guardrails-generation.md`. Two execution options:

1. Subagent-Driven (recommended) - fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
