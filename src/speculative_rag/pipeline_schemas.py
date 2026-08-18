from enum import Enum

from pydantic import BaseModel, Field

from speculative_rag.schemas import Chunk


class RagOutcome(str, Enum):
    ANSWERED = "answered"
    OFF_TOPIC_REFUSED = "off_topic_refused"
    UNSAFE_INPUT_REFUSED = "unsafe_input_refused"
    UNGROUNDED_REFUSED = "ungrounded_refused"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


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
    # The untrimmed sentence `text` was selected/trimmed from -- so
    # groundedness validation and provenance always have the full original
    # sentence available even when a tighter clause is displayed as `text`.
    raw_sentence: str = ""
    # Query tokens matched in raw_sentence, for a future UI pass to
    # highlight matches without reimplementing token matching in
    # JavaScript. Not consumed by app.js in this change.
    matched_tokens: list[str] = Field(default_factory=list)


class StageTiming(BaseModel):
    stage: str
    elapsed_ms: float


class PipelineResult(BaseModel):
    outcome: RagOutcome
    answer: Answer | None = None
    refusal_reason: str | None = None
    top_score: float | None = None
    timings: list[StageTiming] = Field(default_factory=list)
    num_retrieved: int = 0
    num_evidence_candidates: int = 0
    # The full retrieved candidate pool (empty for UNSAFE_INPUT_REFUSED,
    # which returns before it ever retrieves anything). Exposed so
    # downstream recovery code (answer_orchestrator.resolve_answer) can
    # reuse it for rerank, synthesis, and grounding-source lookup instead of
    # paying for a second embed+retrieve round trip against the same query.
    retrieved_chunks: list[ScoredChunk] = Field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(timing.elapsed_ms for timing in self.timings)
