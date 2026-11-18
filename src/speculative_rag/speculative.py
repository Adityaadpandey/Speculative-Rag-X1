"""Speculative RAG drafting and verification (Wang et al. 2024,
arXiv 2407.08223), run over the pool the fast path already retrieved.

The paper's idea, in the order this module does it:

    1. Embed the question and every retrieved passage.
    2. Cluster the passages by content, so each cluster is one "perspective"
       on the question (say: passages about India's borders, passages about
       its states, passages about the Indian Plate).
    3. Build several small evidence subsets, each taking one passage from
       every cluster. Every subset is diverse, and different subsets cover
       different passages, instead of every draft reading the same top 3.
    4. A small, fast drafter model writes an answer plus a short rationale
       for each subset, all in parallel. How sure it was (the average
       probability of the tokens it wrote) is the draft confidence.
    5. A bigger verifier model reads each (answer, rationale) pair and we
       take P("Yes") for "does the rationale support this answer?". The
       draft with the highest draft confidence x support wins.

The verifier never generates text, it only scores, which is what makes this
cheaper than asking the big model to read all twenty passages and write the
answer itself.

One deliberate gap versus the paper: it also multiplies in a
self-consistency score, the verifier's probability of the draft itself.
That needs logprobs over the prompt, which the chat API doesn't return, so
we use the two scores the API can give us.

Everything here runs in the async recovery chain (answer_orchestrator), never
on the <200ms extractive hot path, and talks to APIs only, so the 512MB host
never loads a model.
"""

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel

from speculative_rag.pipeline_schemas import ScoredChunk

DRAFTER_MODEL = os.environ.get("OPENAI_DRAFTER_MODEL", "gpt-5.4-nano")
VERIFIER_MODEL = os.environ.get("OPENAI_VERIFIER_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

NUM_CLUSTERS = 3
NUM_DRAFTS = 4
# Passages are truncated before embedding and drafting. MS MARCO passages
# are short, so this rarely bites; it just caps a pathological one.
_MAX_PASSAGE_CHARS = 2_000
NOT_ANSWERABLE = "NOT ANSWERABLE"


class Draft(BaseModel):
    answer: str
    rationale: str
    source_ids: list[str]  # chunk_ids of the subset this draft read
    draft_confidence: float  # drafter's mean token probability
    support: float  # verifier's P(Yes) that the rationale supports the answer
    score: float  # draft_confidence * support


class SpeculativeResult(BaseModel):
    drafts: list[Draft]
    best: Draft | None
    reasoning: str


# ---------------------------------------------------------------- clustering


def cluster_passages(passage_vecs: np.ndarray, query_vec: np.ndarray, k: int, iterations: int = 10) -> list[int]:
    """k-means on unit vectors (so dot product = cosine). Seeded
    deterministically: the first centre is the passage closest to the query,
    each next one is the passage farthest from every centre so far. Same
    input, same clusters, which keeps the eval reproducible."""
    n = len(passage_vecs)
    k = max(1, min(k, n))
    centres = [int(np.argmax(passage_vecs @ query_vec))]
    while len(centres) < k:
        closest = np.max(passage_vecs @ passage_vecs[centres].T, axis=1)
        centres.append(int(np.argmin(closest)))
    centroids = passage_vecs[centres]
    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        labels = np.argmax(passage_vecs @ centroids.T, axis=1)
        for c in range(k):
            members = passage_vecs[labels == c]
            if len(members):
                mean = members.mean(axis=0)
                centroids[c] = mean / (np.linalg.norm(mean) or 1.0)
    return labels.tolist()


def build_subsets(labels: list[int], query_sims: list[float], num_subsets: int) -> list[list[int]]:
    """Subset j takes, from every cluster, that cluster's j-th most
    query-relevant passage (wrapping around small clusters). So subset 0 is
    the best passage per perspective, subset 1 the runner-up per
    perspective, and so on. Duplicate subsets are dropped."""
    clusters: dict[int, list[int]] = {}
    for idx in sorted(range(len(labels)), key=lambda i: -query_sims[i]):
        clusters.setdefault(labels[idx], []).append(idx)
    subsets: list[list[int]] = []
    for j in range(num_subsets):
        subset = sorted({members[j % len(members)] for members in clusters.values()})
        if subset not in subsets:
            subsets.append(subset)
    return subsets


# ------------------------------------------------------------------ drafting

_DRAFT_PROMPT = """Answer the question using only the evidence below.

Question:
{query_text}

Evidence:
{evidence_block}

Reply in exactly this format, in the same language as the question:
Rationale: <one or two sentences on which evidence supports the answer and why>
Answer: <a concise, complete answer>

If the evidence does not answer the question, reply with
Rationale: <why not>
Answer: {not_answerable}"""

_VERIFY_PROMPT = """Question: {query_text}
Answer: {answer}
Rationale: {rationale}

Does the rationale support the answer, and is the answer a direct, informative reply to the question as asked (not a vague or tangential one)? Reply Yes or No."""
# Wording tested on real drafts: "India is near Punjab" for "where is india"
# scores 0.0 and "Hungary is east of Slovenia" for "where is Hungary" 0.003,
# while correct answers in English, Hindi and Urdu score 1.0. A stricter
# "would a careful reader be fully satisfied" wording rejected correct
# answers too, so it was dropped.

_DRAFT_FORMAT = re.compile(r"Rationale:\s*(?P<rationale>.*?)\s*Answer:\s*(?P<answer>.*)", re.IGNORECASE | re.DOTALL)


def parse_draft(text: str) -> tuple[str, str] | None:
    """(answer, rationale), or None when the drafter declined or ignored the format."""
    match = _DRAFT_FORMAT.search(text)
    if match is None:
        return None
    answer = match["answer"].strip()
    if not answer or NOT_ANSWERABLE.lower() in answer.lower():
        return None
    return answer, match["rationale"].strip()


def _client():
    from speculative_rag.openai_reflect import _get_client

    return _get_client()


def openai_embed(texts: list[str]) -> np.ndarray:
    response = _client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vecs = np.array([item.embedding for item in response.data], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def openai_draft(query_text: str, evidence: list[str]) -> tuple[str, float]:
    """Returns the drafter's raw text and its mean token probability."""
    evidence_block = "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(evidence))
    response = _client().chat.completions.create(
        model=DRAFTER_MODEL,
        messages=[{"role": "user", "content": _DRAFT_PROMPT.format(
            query_text=query_text, evidence_block=evidence_block, not_answerable=NOT_ANSWERABLE
        )}],
        logprobs=True,
    )
    choice = response.choices[0]
    logprobs = [t.logprob for t in (choice.logprobs.content or [])] if choice.logprobs else []
    # Geometric mean, so a long careful draft isn't punished for its length.
    confidence = math.exp(sum(logprobs) / len(logprobs)) if logprobs else 0.5
    return choice.message.content or "", confidence


def openai_verify(query_text: str, answer: str, rationale: str) -> float:
    """P(Yes) / (P(Yes) + P(No)) from the verifier's first output token."""
    response = _client().chat.completions.create(
        model=VERIFIER_MODEL,
        messages=[{"role": "user", "content": _VERIFY_PROMPT.format(
            query_text=query_text, answer=answer, rationale=rationale
        )}],
        logprobs=True,
        top_logprobs=10,
        max_completion_tokens=1,
    )
    top = response.choices[0].logprobs.content[0].top_logprobs
    yes = sum(math.exp(t.logprob) for t in top if t.token.strip().lower() == "yes")
    no = sum(math.exp(t.logprob) for t in top if t.token.strip().lower() == "no")
    return yes / (yes + no) if yes + no else 0.0


# ---------------------------------------------------------------- the whole loop


def run_speculative_rag(
    query_text: str,
    candidates: list[ScoredChunk],
    on_stage=None,
    *,
    embed=openai_embed,
    draft=openai_draft,
    verify=openai_verify,
    num_clusters: int = NUM_CLUSTERS,
    num_drafts: int = NUM_DRAFTS,
) -> SpeculativeResult:
    on_stage = on_stage or (lambda stage: None)
    if not candidates:
        return SpeculativeResult(drafts=[], best=None, reasoning="no candidates retrieved")

    texts = [c.chunk.text[:_MAX_PASSAGE_CHARS] for c in candidates]
    on_stage("drafting")
    vecs = embed([query_text, *texts])
    query_vec, passage_vecs = vecs[0], vecs[1:]
    query_sims = (passage_vecs @ query_vec).tolist()
    labels = cluster_passages(passage_vecs, query_vec, num_clusters)
    subsets = build_subsets(labels, query_sims, num_drafts)

    def make_draft(subset: list[int]) -> Draft | None:
        raw, confidence = draft(query_text, [texts[i] for i in subset])
        parsed = parse_draft(raw)
        if parsed is None:
            return None
        answer, rationale = parsed
        ids = [candidates[i].chunk.chunk_id for i in subset]
        return Draft(answer=answer, rationale=rationale, source_ids=ids,
                     draft_confidence=confidence, support=0.0, score=0.0)

    with ThreadPoolExecutor(max_workers=len(subsets)) as pool:
        drafts = [d for d in pool.map(make_draft, subsets) if d is not None]
    if not drafts:
        return SpeculativeResult(
            drafts=[], best=None,
            reasoning=f"none of the {len(subsets)} drafts found an answer in its evidence subset",
        )

    on_stage("verifying_drafts")
    with ThreadPoolExecutor(max_workers=len(drafts)) as pool:
        supports = list(pool.map(lambda d: verify(query_text, d.answer, d.rationale), drafts))
    drafts = [
        d.model_copy(update={"support": s, "score": d.draft_confidence * s})
        for d, s in zip(drafts, supports)
    ]
    drafts.sort(key=lambda d: d.score, reverse=True)
    best = drafts[0]
    return SpeculativeResult(
        drafts=drafts, best=best,
        reasoning=f"picked 1 of {len(drafts)} drafts (score {best.score:.2f}): {best.rationale}",
    )
