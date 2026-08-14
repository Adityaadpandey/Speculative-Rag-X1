"""Async LLM-based groundedness check (spec: rho_Self-reflect). Purely
additive, non-hot-path: never imported by pipeline.py, never consulted
inside run_pipeline, never changes RagOutcome. A secondary, informational
verdict on whether an already-returned ANSWERED result is actually
grounded in its source passage, using Claude Sonnet 5 structured output."""

import anthropic
from pydantic import BaseModel

_MODEL = "claude-sonnet-5"

_PROMPT_TEMPLATE = """You are reviewing whether an answer is grounded in its source passage.

Source passage:
{source_text}

Answer:
{answer_text}

Decide whether the answer is fully supported by the source passage (grounded=true) \
or introduces claims not present in the source passage (grounded=false). Give a \
brief one or two sentence reasoning for your verdict."""


class ReflectVerdict(BaseModel):
    grounded: bool
    reasoning: str


def check_groundedness_llm(answer_text: str, source_text: str) -> ReflectVerdict:
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": _PROMPT_TEMPLATE.format(
                    source_text=source_text, answer_text=answer_text
                ),
            }
        ],
        output_format=ReflectVerdict,
    )
    verdict = response.parsed_output
    assert verdict is not None
    return verdict
