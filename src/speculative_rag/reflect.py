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
its source passage. Evaluate FOUR separate questions independently — do not let one verdict influence \
another. A verbatim, fully-supported excerpt from the source passage can still fail multiple of these.

User query:
{query_text}

Source passage:
{source_text}

Answer:
{answer_text}

1. grounded (faithfulness): Is every claim in the answer supported by the source passage, with no \
claims introduced that aren't present in the source passage?

2. answers_query (answerability): Does the answer directly and specifically answer the user's query — \
stating the specific fact, entity, date, number, or yes/no verdict the query is actually asking for? \
A verbatim excerpt that is merely on-topic but does NOT state the specific thing the query asks for \
must be scored answers_query=false, even though it may still be grounded=true.

3. complete (completeness): Does the answer contain the important information required to fully \
answer the query, or is it missing a necessary part of the answer?

4. relevant (relevance): Is the answer focused on the query, rather than mostly consisting of \
surrounding context, tangents, or information unrelated to what was asked?

These four verdicts are independent: an answer can be grounded but not answer the query, can answer \
the query but include an unsupported claim, can answer the query yet be incomplete, or any other \
combination — including all pass or all fail.

Give a brief one or two sentence reasoning covering all four verdicts."""


class ReflectVerdict(BaseModel):
    grounded: bool
    answers_query: bool
    complete: bool
    relevant: bool
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


# Async multi-passage synthesis: when extractive top-1 selection fails
# (select_answer_span returned None, or the async reflect verdict says the
# top-1 answer is incomplete or doesn't answer the query), a single passage
# might still not have the full answer even though the top-k collectively
# does. This makes one more LLM call across up to k candidate passages,
# asking it to synthesize -- not copy -- a concise answer only if the
# evidence actually supports one, and to say so explicitly (never invent
# from world knowledge) when it doesn't. Non-hot-path, same pattern as the
# two functions above.
_SYNTHESIS_PROMPT_TEMPLATE = """You are answering a user's question using ONLY the evidence in the \
candidate passages below. Do not use outside knowledge.

Question:
{query_text}

Candidate passages:
{candidates_block}

First decide: do these passages, taken together, contain enough information to answer the question?

If yes: write a concise answer (one or two sentences) that states the specific fact the question asks \
for. Synthesize from the relevant parts of the passages — do not copy an entire passage verbatim, and \
do not include information irrelevant to the question. List the 0-based indices of the passages you \
actually used.

If no: do not guess or invent an answer from general knowledge. Set answerable=false, leave \
answer_text empty, and briefly explain what information is missing."""


class SynthesisVerdict(BaseModel):
    answerable: bool
    answer_text: str
    source_indices: list[int]
    reasoning: str


def synthesize_answer_llm(query_text: str, candidate_texts: list[str]) -> SynthesisVerdict:
    client = anthropic.Anthropic()
    candidates_block = "\n".join(f"[{i}] {text}" for i, text in enumerate(candidate_texts))
    response = client.messages.parse(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": _SYNTHESIS_PROMPT_TEMPLATE.format(
                    query_text=query_text, candidates_block=candidates_block
                ),
            }
        ],
        output_format=SynthesisVerdict,
    )
    verdict = response.parsed_output
    assert verdict is not None
    verdict.source_indices = [i for i in verdict.source_indices if 0 <= i < len(candidate_texts)]
    if not verdict.answerable:
        verdict.answer_text = ""
        verdict.source_indices = []
    return verdict
