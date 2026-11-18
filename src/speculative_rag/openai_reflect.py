"""OpenAI-backed LLM recovery-chain calls: groundedness check, passage
rerank, rerank+generate, multi-passage synthesis, and streamed answer
generation. Same prompts and same parsed return shapes as reflect.py (the
Anthropic-backed original) -- only the provider differs, so this is a
drop-in replacement for every function reflect.py exposes to
answer_orchestrator.py and webapp/main.py.

reflect.py itself is left alone (imports Anthropic, keeps its own test
suite green) rather than rewritten in place, so this file, not reflect.py,
is what production and the eval harness both point at when running on
OPENAI_API_KEY.
"""
import os

import openai

from speculative_rag.reflect import (
    RerankAndAnswer,
    RerankCandidate,
    RerankVerdict,
    ReflectVerdict,
    SelectedSource,
    SynthesisVerdict,
    _PROMPT_TEMPLATE,
    _RERANK_AND_GENERATE_PROMPT_TEMPLATE,
    _RERANK_PROMPT_TEMPLATE,
    _RerankAndGenerateParsed,
    _RerankParsed,
    _SYNTHESIS_PROMPT_TEMPLATE,
    _GENERATE_ANSWER_PROMPT_TEMPLATE,
    strip_answerable_marker,
)

MODEL = os.environ.get("OPENAI_GENERATION_MODEL", "gpt-5.6-luna")

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client


def check_groundedness(query_text: str, answer_text: str, source_text: str) -> ReflectVerdict:
    client = _get_client()
    response = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": _PROMPT_TEMPLATE.format(
                    query_text=query_text, source_text=source_text, answer_text=answer_text
                ),
            }
        ],
        response_format=ReflectVerdict,
    )
    parsed = response.choices[0].message.parsed
    assert parsed is not None
    return parsed


def rerank_passages(query_text: str, candidates: list[RerankCandidate]) -> RerankVerdict:
    client = _get_client()
    candidates_block = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(candidates))
    response = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": _RERANK_PROMPT_TEMPLATE.format(
                    query_text=query_text, candidates_block=candidates_block
                ),
            }
        ],
        response_format=_RerankParsed,
    )
    parsed = response.choices[0].message.parsed
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


def rerank_and_generate(query_text: str, candidates: list[RerankCandidate]) -> RerankAndAnswer:
    client = _get_client()
    candidates_block = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(candidates))
    response = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": _RERANK_AND_GENERATE_PROMPT_TEMPLATE.format(
                    query_text=query_text, candidates_block=candidates_block
                ),
            }
        ],
        response_format=_RerankAndGenerateParsed,
    )
    parsed = response.choices[0].message.parsed
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


def synthesize_answer(query_text: str, candidate_texts: list[str]) -> SynthesisVerdict:
    client = _get_client()
    candidates_block = "\n".join(f"[{i}] {text}" for i, text in enumerate(candidate_texts))
    response = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": _SYNTHESIS_PROMPT_TEMPLATE.format(
                    query_text=query_text, candidates_block=candidates_block
                ),
            }
        ],
        response_format=SynthesisVerdict,
    )
    verdict = response.choices[0].message.parsed
    assert verdict is not None
    verdict.source_indices = [i for i in verdict.source_indices if 0 <= i < len(candidate_texts)]
    if not verdict.answerable:
        verdict.answer_text = ""
        verdict.source_indices = []
    return verdict


def stream_generate_answer(query_text: str, evidence_texts: list[str], on_delta) -> str:
    """Stream the answer text while keeping final validation synchronous --
    same contract as reflect.stream_generate_answer_llm: only the prose
    answer is streamed, the completed text still goes through the normal
    groundedness check before it becomes final."""
    client = _get_client()
    evidence_block = "\n".join(f"- {text}" for text in evidence_texts)
    prompt = _GENERATE_ANSWER_PROMPT_TEMPLATE.format(query_text=query_text, evidence_block=evidence_block)
    parts: list[str] = []
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
            on_delta(delta)
    return strip_answerable_marker("".join(parts))
