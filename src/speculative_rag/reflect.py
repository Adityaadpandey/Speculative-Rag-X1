"""Async LLM-based groundedness check. Purely additive, non-hot-path: never
imported by pipeline.py, never consulted inside run_pipeline, never changes
RagOutcome. It's a secondary, informational verdict on whether an
already-returned ANSWERED result is actually grounded in its source
passage, using Claude Haiku structured output."""

import re

import anthropic
from pydantic import BaseModel

import speculative_rag.config  # noqa: F401  # side effect: loads .env so ANTHROPIC_API_KEY is set before anthropic.Anthropic() reads it

_MODEL = "claude-haiku-4-5-20251001"


def _anthropic_client():
    # Keep a bounded production timeout. The fallback call makes the helper
    # compatible with lightweight test doubles that accept no constructor
    # arguments.
    try:
        return anthropic.Anthropic(timeout=30.0)
    except TypeError:
        return anthropic.Anthropic()

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

For completeness, do not accept a bare restatement such as "India is a peninsula" when the source \
also explains what that means or names the surrounding seas. For "how", "why", and "what does X \
mean" questions, the answer must include the explanatory relationship and key supporting details \
present in the source.

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
    client = _anthropic_client()
    response = client.messages.parse(
        model=_MODEL,
        max_tokens=2048,
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
#
# This returns structured selections (source id + relevance score), never
# prose explanations of the pick, and it has to be able to select nothing at
# all: when none of the candidates actually answer the question,
# answerable=false and selected_sources=[] rather than forcing a
# best-of-a-bad-lot guess -- a forced guess is exactly what let the
# downstream generator invent an answer from an irrelevant passage before.
_RERANK_PROMPT_TEMPLATE = """You are selecting which candidate passages, if any, contain the evidence \
needed to answer a question. Do not answer the question yourself -- only judge relevance.

Question:
{query_text}

Candidate passages:
{candidates_block}

For every candidate that is actually relevant to answering the question, report its 0-based index and \
a relevance score from 0.0 to 1.0. Do not include candidates that are merely on-topic but do not bear on \
the answer. A passage about the same general subject that never states the specific fact, entity, \
distinction, number, or comparison the question asks for is NOT relevant -- score that the same as an \
unrelated passage, not as a partial match.

If NONE of the candidates contain information that answers the question, return an empty selection and \
set answerable to false -- do not force a selection just because candidates exist. When genuinely unsure \
whether a candidate answers the question or merely resembles it, decline rather than guess.

Give a brief one-sentence reasoning."""


class _RerankSelection(BaseModel):
    index: int
    relevance_score: float


class _RerankParsed(BaseModel):
    selections: list[_RerankSelection]
    answerable: bool
    reasoning: str


class RerankCandidate(BaseModel):
    source_id: str
    text: str


class SelectedSource(BaseModel):
    source_id: str
    relevance_score: float


class RerankVerdict(BaseModel):
    selected_sources: list[SelectedSource]
    answerable: bool
    reasoning: str


def rerank_passages_llm(query_text: str, candidates: list[RerankCandidate]) -> RerankVerdict:
    client = _anthropic_client()
    candidates_block = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(candidates))
    response = client.messages.parse(
        model=_MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": _RERANK_PROMPT_TEMPLATE.format(
                    query_text=query_text, candidates_block=candidates_block
                ),
            }
        ],
        output_format=_RerankParsed,
    )
    parsed = response.parsed_output
    assert parsed is not None

    selected_sources = [
        SelectedSource(source_id=candidates[s.index].source_id, relevance_score=s.relevance_score)
        for s in parsed.selections
        if 0 <= s.index < len(candidates)
    ]
    answerable = parsed.answerable and bool(selected_sources)
    if not answerable:
        selected_sources = []
    return RerankVerdict(selected_sources=selected_sources, answerable=answerable, reasoning=parsed.reasoning)


# Real answer generation from evidence, used to regenerate the final answer
# once rerank_passages_llm has found the right passage(s) -- the step the
# fast extractive hot path can't do (it can only return a verbatim span,
# which is why an on-topic-but-non-answering rank-1 passage produces a bad
# "answer" with groundedness ~1.0 by construction). This is a real
# instruction-following generation call: answer the question, not
# "summarize this passage" -- and it needs to be able to say the evidence is
# insufficient rather than inventing something, same never-hallucinate
# contract as synthesize_answer_llm below.
_GENERATE_ANSWER_PROMPT_TEMPLATE = """Question:
{query_text}

Evidence:
{evidence_block}

Instructions:
Answer the question directly using only the evidence above.
Do not add unsupported facts, and do not invent anything not present in the evidence.
Do not answer a different but related question -- answer exactly what was asked.
For "how", "why", and definition questions, do not merely repeat the premise. Include the explanation
and key supporting details stated in the evidence (for example, named boundaries or causes).
Evidence about the same general subject that never states the specific fact, entity, distinction,
number, or comparison asked for does not answer the question -- treat that the same as having no
evidence at all. If the evidence does not answer the question, say so explicitly and set answerable
to false rather than offering the closest related thing you can find.
Keep the answer concise but complete (one or two sentences)."""


class GeneratedAnswer(BaseModel):
    answerable: bool
    answer_text: str
    reasoning: str


def generate_answer_llm(query_text: str, evidence_texts: list[str]) -> GeneratedAnswer:
    client = _anthropic_client()
    evidence_block = "\n".join(f"- {text}" for text in evidence_texts)
    response = client.messages.parse(
        model=_MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": _GENERATE_ANSWER_PROMPT_TEMPLATE.format(
                    query_text=query_text, evidence_block=evidence_block
                ),
            }
        ],
        output_format=GeneratedAnswer,
    )
    parsed = response.parsed_output
    assert parsed is not None
    if not parsed.answerable:
        parsed.answer_text = ""
    return parsed


# The generation prompt above says "set answerable to false", which makes
# sense for the structured call but not for the streamed one: there's no
# field to set, so the model writes the flag into the prose instead, e.g.
#   "India is located in South Asia.\n\nanswerable: true"
#   "The evidence does not say where India is.\n**Answerable:** false"
_ANSWERABLE_MARKER = re.compile(r"(?:^|\n)[^\w\n]*answerable\W*:\W*(true|false)\W*$", re.IGNORECASE)


def strip_answerable_marker(text: str) -> str:
    """Drop a trailing "answerable: true/false" line from streamed prose.
    A "false" marker means the model is declining, so we return "" and the
    orchestrator treats that as no answer and escalates to synthesis
    instead of verifying the refusal sentence as if it were an answer."""
    match = _ANSWERABLE_MARKER.search(text)
    if match is None:
        return text.strip()
    if match.group(1).lower() == "false":
        return ""
    return text[: match.start()].strip()


def stream_generate_answer_llm(
    query_text: str,
    evidence_texts: list[str],
    on_delta,
) -> str:
    """Stream the answer text while keeping final validation synchronous.

    Passage selection remains structured and completes first. Only the prose
    answer is streamed to the UI; the completed text is still validated by
    the normal groundedness/relevance checks before it becomes final.
    """
    client = _anthropic_client()
    evidence_block = "\n".join(f"- {text}" for text in evidence_texts)
    prompt = _GENERATE_ANSWER_PROMPT_TEMPLATE.format(
        query_text=query_text,
        evidence_block=evidence_block,
    )
    parts: list[str] = []
    with client.messages.stream(
        model=_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            parts.append(text)
            on_delta(text)
    return strip_answerable_marker("".join(parts))


# Latency optimization for answer_orchestrator.resolve_recovery:
# rerank_passages_llm and generate_answer_llm above are two separate network
# round trips doing "which passage is right" then "answer from it" -- this
# combines both into one call, the same pattern synthesize_answer_llm below
# already uses (select sources + answer in a single structured response).
# Cuts one full round trip out of the common recovery path.
# check_groundedness_llm stays a fully separate, independent verification
# call either way -- only the selection and generation steps merge, never
# the verification of them.
_RERANK_AND_GENERATE_PROMPT_TEMPLATE = """You are given a question and a pool of candidate passages. In ONE pass: \
(1) decide which candidate passages, if any, actually contain the evidence needed to answer the question, and \
(2) if any do, answer the question directly using ONLY that evidence.

Question:
{query_text}

Candidate passages:
{candidates_block}

Step 1 -- selection: for every candidate that is actually relevant to answering the question, report its 0-based \
index and a relevance score from 0.0 to 1.0. Do not include candidates that are merely on-topic but do not bear \
on the answer. A passage about the same general subject that never states the specific fact, entity, \
distinction, number, or comparison the question asks for is NOT relevant -- score it the same as an unrelated \
passage, not as a partial match. If NONE of the candidates contain information that answers the question, \
return an empty selection and set selection_found to false -- do not force a selection just because candidates \
exist. When genuinely unsure whether a candidate answers the question or merely resembles it, decline rather \
than guess.

Step 2 -- answer: if step 1 found relevant evidence, answer the question directly using only that evidence. Do \
not add unsupported facts, do not invent anything not present in the evidence, and do not answer a different but \
related question -- answer exactly what was asked. For "how", "why", and definition questions, do not \
merely repeat the premise; include the explanation and key supporting details in the selected evidence. \
Keep the answer concise but complete (one or two sentences). If the selected evidence turns out not to \
actually answer the question, or step 1 found nothing, set answerable to false and leave answer_text empty \
rather than guessing.

Give a brief one-sentence reasoning covering both the selection and the answer."""


class _RerankAndGenerateSelection(BaseModel):
    index: int
    relevance_score: float


class _RerankAndGenerateParsed(BaseModel):
    selections: list[_RerankAndGenerateSelection]
    selection_found: bool
    answerable: bool
    answer_text: str
    reasoning: str


class RerankAndAnswer(BaseModel):
    selected_sources: list[SelectedSource]
    selection_found: bool
    answerable: bool
    answer_text: str
    reasoning: str


def rerank_and_generate_llm(query_text: str, candidates: list[RerankCandidate]) -> RerankAndAnswer:
    client = _anthropic_client()
    candidates_block = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(candidates))
    response = client.messages.parse(
        model=_MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": _RERANK_AND_GENERATE_PROMPT_TEMPLATE.format(
                    query_text=query_text, candidates_block=candidates_block
                ),
            }
        ],
        output_format=_RerankAndGenerateParsed,
    )
    parsed = response.parsed_output
    assert parsed is not None

    selected_sources = [
        SelectedSource(source_id=candidates[s.index].source_id, relevance_score=s.relevance_score)
        for s in parsed.selections
        if 0 <= s.index < len(candidates)
    ]
    selection_found = parsed.selection_found and bool(selected_sources)
    if not selection_found:
        selected_sources = []
    answerable = selection_found and parsed.answerable and bool(parsed.answer_text)
    return RerankAndAnswer(
        selected_sources=selected_sources,
        selection_found=selection_found,
        answerable=answerable,
        answer_text=parsed.answer_text if answerable else "",
        reasoning=parsed.reasoning,
    )


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

First decide: do these passages, taken together, contain enough information to answer the question? \
Passages about the same general subject that never state the specific fact, entity, distinction, \
number, or comparison asked for do NOT count -- that's the same as having no relevant evidence at all, \
even if the passages are on-topic.

If yes: write a concise answer (one or two sentences) that states the specific fact the question asks \
for. Synthesize from the relevant parts of the passages — do not copy an entire passage verbatim, and \
do not include information irrelevant to the question. For "how", "why", and definition questions, \
include the explanation and key supporting details rather than only restating the premise. List the \
0-based indices of the passages you actually used.

If no, or if you are genuinely unsure whether the passages actually answer the question rather than \
merely resembling it: do not guess or invent an answer from general knowledge. Set answerable=false, \
leave answer_text empty, and briefly explain what information is missing."""


class SynthesisVerdict(BaseModel):
    answerable: bool
    answer_text: str
    source_indices: list[int]
    reasoning: str


def synthesize_answer_llm(query_text: str, candidate_texts: list[str]) -> SynthesisVerdict:
    client = _anthropic_client()
    candidates_block = "\n".join(f"[{i}] {text}" for i, text in enumerate(candidate_texts))
    response = client.messages.parse(
        model=_MODEL,
        max_tokens=2048,
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
