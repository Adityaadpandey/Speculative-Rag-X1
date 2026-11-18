import numpy as np

from speculative_rag.answer_orchestrator import resolve_recovery
from speculative_rag.pipeline_schemas import PipelineResult, RagOutcome, ScoredChunk
from speculative_rag.reflect import ReflectVerdict
from speculative_rag.schemas import Chunk
from speculative_rag.speculative import build_subsets, cluster_passages, parse_draft, run_speculative_rag


def _chunk(i: int, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=f"c{i}", text=text, language="en", query_type="LOCATION",
                    source_passage_id=f"p{i}", strategy="native"),
        score=10.0 - i,
    )


# Three obvious "perspectives" in 2D: passages 0-1 point east, 2-3 north, 4-5 west.
_VECS = {
    "east a": [1.0, 0.0], "east b": [0.95, 0.05],
    "north a": [0.0, 1.0], "north b": [0.05, 0.95],
    "west a": [-1.0, 0.0], "west b": [-0.95, 0.05],
}


def _fake_embed(texts):
    vecs = np.array([[0.9, 0.1]] + [_VECS[t] for t in texts[1:]], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_clusters_split_by_direction_and_each_subset_spans_every_cluster():
    vecs = _fake_embed(["q", *_VECS])
    labels = cluster_passages(vecs[1:], vecs[0], k=3)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[4] == labels[5]
    assert len({labels[0], labels[2], labels[4]}) == 3

    sims = (vecs[1:] @ vecs[0]).tolist()
    subsets = build_subsets(labels, sims, num_subsets=4)
    # Two members per cluster -> only two distinct subsets exist; together
    # they read every passage, and each spans all three perspectives.
    assert len(subsets) == 2
    assert sorted(i for s in subsets for i in s) == [0, 1, 2, 3, 4, 5]
    for subset in subsets:
        assert len({labels[i] for i in subset}) == 3
    # The first subset holds each cluster's most query-relevant passage.
    for i in subsets[0]:
        assert all(sims[i] >= sims[j] for j in range(6) if labels[j] == labels[i])


def test_parse_draft_reads_format_and_drops_refusals():
    assert parse_draft("Rationale: [1] says so.\nAnswer: South Asia.") == ("South Asia.", "[1] says so.")
    assert parse_draft("Rationale: nothing relevant\nAnswer: NOT ANSWERABLE") is None
    assert parse_draft("India is in Asia.") is None


def test_verifier_score_picks_the_winner_not_draft_order():
    candidates = [_chunk(i, t) for i, t in enumerate(_VECS)]

    def draft(query, evidence):
        if "east a" in evidence:
            return "Rationale: weak guess\nAnswer: Europe", 0.9
        return "Rationale: [2] states it directly\nAnswer: South Asia", 0.8

    def verify(query, answer, rationale):
        return 0.2 if answer == "Europe" else 0.95

    result = run_speculative_rag("where is india", candidates, embed=_fake_embed, draft=draft, verify=verify)
    assert result.best.answer == "South Asia"
    assert result.best.score == 0.8 * 0.95
    assert [d.score for d in result.drafts] == sorted((d.score for d in result.drafts), reverse=True)


def test_all_drafts_declining_means_no_answer():
    candidates = [_chunk(i, t) for i, t in enumerate(_VECS)]
    result = run_speculative_rag(
        "q", candidates, embed=_fake_embed,
        draft=lambda q, e: ("Rationale: no\nAnswer: NOT ANSWERABLE", 0.9),
        verify=lambda *a: 1.0,
    )
    assert result.best is None and result.drafts == []


def test_orchestrator_returns_verified_speculative_draft():
    candidates = [_chunk(i, t) for i, t in enumerate(_VECS)]
    pipeline = PipelineResult(outcome=RagOutcome.INSUFFICIENT_EVIDENCE, retrieved_chunks=candidates,
                              timings=[], total_ms=1.0)

    def speculative(query, cands, on_stage):
        return run_speculative_rag(
            query, cands, on_stage, embed=_fake_embed,
            draft=lambda q, e: ("Rationale: [1] says so\nAnswer: South Asia", 0.9),
            verify=lambda *a: 0.9,
        )

    grounded = ReflectVerdict(grounded=True, answers_query=True, complete=True, relevant=True, reasoning="ok")
    outcome = resolve_recovery("where is india", pipeline, check_groundedness=lambda *a: grounded,
                               speculative=speculative)
    assert outcome.final_status.value == "answered"
    assert outcome.answer_engine == "speculative_draft"
    assert outcome.answer_text == "South Asia"
    assert outcome.drafts and outcome.drafts[0].score == 0.81
