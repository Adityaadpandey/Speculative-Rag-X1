"""Async LLM-based groundedness check (spec: rho_Self-reflect). Purely
additive, non-hot-path: never imported by pipeline.py, never consulted
inside run_pipeline, never changes RagOutcome. A secondary, informational
verdict on whether an already-returned ANSWERED result is actually
grounded in its source passage, using Claude Sonnet 5 structured output."""

import anthropic
from pydantic import BaseModel

import speculative_rag.config  # noqa: F401  # side effect: loads .env so ANTHROPIC_API_KEY is set before anthropic.Anthropic() reads it

_MODEL = "claude-sonnet-5"

_PROMPT_TEMPLATE = """You are reviewing an answer produced for a user's query, checking it against \
its source passage. Evaluate TWO separate questions independently — do not conflate them.

User query:
{query_text}

Source passage:
{source_text}

Answer:
{answer_text}

1. grounded: Is every claim in the answer supported by the source passage, with no claims \
introduced that aren't present in the source passage?

2. answers_query: Does the answer directly and specifically answer the user's query — stating \
the specific fact, entity, date, number, or yes/no verdict the query is actually asking for? \
A verbatim, fully-supported excerpt from the source passage that is merely on-topic but does \
NOT state the specific thing the query asks for must be scored answers_query=false, even though \
it may still be grounded=true. grounded and answers_query are independent verdicts: an answer \
can be grounded but not answer the query, can answer the query but include an unsupported \
claim, both, or neither.

Give a brief one or two sentence reasoning covering both verdicts."""


class ReflectVerdict(BaseModel):
    grounded: bool
    answers_query: bool
    reasoning: str


def check_groundedness_llm(query_text: str, answer_text: str, source_text: str) -> ReflectVerdict:
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": _PROMPT_TEMPLATE.format(
                    query_text=query_text, source_text=source_text, answer_text=answer_text
                ),
            }
        ],
        output_format=ReflectVerdict,
    )
    verdict = response.parsed_output
    assert verdict is not None
    return verdict


# Async LLM reranker: a second, non-hot-path quality layer for the OTHER
# problem the groundedness check above can't see. Groundedness checks "is
# the answer faithful to the passage it was extracted from" -- it says
# nothing about whether that was the right passage to begin with. Embedding
# rank-1 alone only picks the correct passage about 24-45% of the time on
# this corpus, and cheap lexical reranking actually made it worse, not
# better: the top-k candidates are frequently near-duplicate passages on the
# same topic, and the dataset's labeled "gold" passage often paraphrases the
# query differently than the verbatim-matching distractors do, so
# bag-of-words overlap systematically favors the wrong one. An LLM judging
# semantic relevance across the candidates, instead of surface overlap, is
# the fix -- reported as its own latency number, never counted against the
# hot-path latency budget.
_RERANK_PROMPT_TEMPLATE = """You are choosing which candidate passage best answers a question.

Question:
{query_text}

Candidate passages:
{candidates_block}

Reply with the 0-based index of the candidate that most directly and correctly answers the \
question, and a brief one or two sentence reasoning."""


class RerankVerdict(BaseModel):
    best_index: int
    reasoning: str


def select_best_candidate_llm(query_text: str, candidate_texts: list[str]) -> RerankVerdict:
    client = anthropic.Anthropic()
    candidates_block = "\n".join(f"[{i}] {text}" for i, text in enumerate(candidate_texts))
    response = client.messages.parse(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": _RERANK_PROMPT_TEMPLATE.format(
                    query_text=query_text, candidates_block=candidates_block
                ),
            }
        ],
        output_format=RerankVerdict,
    )
    verdict = response.parsed_output
    assert verdict is not None
    if not (0 <= verdict.best_index < len(candidate_texts)):
        verdict.best_index = 0
    return verdict
