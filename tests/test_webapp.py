from fastapi.testclient import TestClient

import webapp.main as webapp_main
from speculative_rag.pipeline_schemas import RetrievalResult, ScoredChunk
from speculative_rag.schemas import Chunk
from webapp.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_get_index_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_collections_returns_parsed_list_and_key_flags(monkeypatch):
    monkeypatch.setattr(
        webapp_main,
        "_COLLECTIONS",
        [{"language": "hi", "strategy": "native"}, {"language": "en", "strategy": "sliding_window"}],
    )
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.get("/api/collections")
    assert response.status_code == 200
    data = response.json()
    assert data["collections"] == [
        {"language": "hi", "strategy": "native"},
        {"language": "en", "strategy": "sliding_window"},
    ]
    assert data["sarvam_key_configured"] is True
    assert data["openai_key_configured"] is False


def test_parse_collection_name_recognizes_known_language_strategy_pairs():
    assert webapp_main._parse_collection_name("hi_native") == {"language": "hi", "strategy": "native"}
    assert webapp_main._parse_collection_name("ta_sliding_window") == {
        "language": "ta",
        "strategy": "sliding_window",
    }
    assert webapp_main._parse_collection_name("unknown_collection") is None


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeRetriever:
    def __init__(self, result: RetrievalResult):
        self.result = result

    def search(self, query_vector, k):
        return self.result


class RaisingRetriever:
    def search(self, query_vector, k):
        raise RuntimeError("boom")


def _scored(text, score, pid="p1"):
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"{pid}:native:0",
            text=text,
            language="hi",
            query_type="DESCRIPTION",
            source_passage_id=pid,
            strategy="native",
        ),
        score=score,
    )


def test_post_query_returns_answered_shape(monkeypatch):
    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9)]))
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    response = client.post(
        "/api/query", json={"query_text": "what is a corporation", "language": "hi", "strategy": "native"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["outcome"] == "answered"
    assert data["answer"]["source_passage_id"] == "p1"
    assert data["refusal_reason"] is None
    assert data["total_ms"] >= 0.0
    assert [t["stage"] for t in data["timings"]] == [
        "guardrail_unsafe",
        "embed",
        "retrieve",
        "guardrail_offtopic",
        "generate",
        "guardrail_groundedness",
        "guardrail_answer_type",
    ]


def test_post_query_returns_refused_shape_with_null_answer(monkeypatch):
    # Under BM25, score=0.0 (no shared vocabulary at all) is the only
    # reliable off-topic signal -- see guardrails.is_off_topic.
    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("something unrelated", 0.0)]))
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    response = client.post(
        "/api/query", json={"query_text": "what is a corporation", "language": "hi", "strategy": "native"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["outcome"] == "off_topic_refused"
    assert data["answer"] is None
    assert data["refusal_reason"] is not None


def test_post_query_unknown_collection_returns_404(monkeypatch):
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {})
    response = client.post("/api/query", json={"query_text": "x", "language": "hi", "strategy": "native"})
    assert response.status_code == 404
    assert "error" in response.json()


def test_post_query_unhandled_exception_returns_500_json(monkeypatch):
    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": RaisingRetriever()})

    response = client.post("/api/query", json={"query_text": "x", "language": "hi", "strategy": "native"})
    assert response.status_code == 500
    assert response.json() == {"error": "boom"}


def test_post_transcribe_returns_transcript_and_latency(monkeypatch):
    def fake_transcribe_audio(path, language_code):
        assert language_code == "hi-IN"
        assert path.exists()
        return "नमस्ते"

    monkeypatch.setattr(webapp_main, "transcribe_audio", fake_transcribe_audio)

    response = client.post(
        "/api/transcribe",
        files={"file": ("clip.wav", b"RIFF....", "audio/wav")},
        data={"language": "hi"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["transcript"] == "नमस्ते"
    assert data["stt_latency_ms"] >= 0.0


def test_post_transcribe_missing_api_key_returns_400(monkeypatch):
    from speculative_rag.stt import SarvamAPIKeyMissingError

    def fake_transcribe_audio(path, language_code):
        raise SarvamAPIKeyMissingError()

    monkeypatch.setattr(webapp_main, "transcribe_audio", fake_transcribe_audio)

    response = client.post(
        "/api/transcribe",
        files={"file": ("clip.wav", b"RIFF....", "audio/wav")},
        data={"language": "hi"},
    )
    assert response.status_code == 400
    assert "SARVAM_API_KEY" in response.json()["error"]


def test_post_reflect_returns_grounded_verdict_shape(monkeypatch):
    from speculative_rag.reflect import ReflectVerdict

    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9)]))
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    captured = {}

    def fake_check_groundedness_llm(query_text, answer_text, source_text):
        captured["query_text"] = query_text
        captured["answer_text"] = answer_text
        captured["source_text"] = source_text
        return ReflectVerdict(
            grounded=True,
            answers_query=True,
            complete=True,
            relevant=True,
            reasoning="Matches the source directly.",
        )

    monkeypatch.setattr(webapp_main, "check_groundedness", fake_check_groundedness_llm)

    response = client.post(
        "/api/reflect",
        json={
            "query_text": "what is a corporation",
            "language": "hi",
            "strategy": "native",
            "answer_text": "A corporation is a company.",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert data["answers_query"] is True
    assert data["complete"] is True
    assert data["relevant"] is True
    assert data["final_status"] == "answered"
    assert data["reasoning"] == "Matches the source directly."
    assert data["reflect_latency_ms"] >= 0.0
    assert captured["query_text"] == "what is a corporation"
    assert captured["answer_text"] == "A corporation is a company."
    assert captured["source_text"] == "A corporation is a company."


def test_post_reflect_ungrounded_verdict_maps_to_not_answered_family(monkeypatch):
    from speculative_rag.reflect import ReflectVerdict

    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    retriever = FakeRetriever(
        RetrievalResult(scored_chunks=[_scored("A corporation is a company.", 0.9, pid="p1")])
    )
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    def fake_check_groundedness_llm(query_text, answer_text, source_text):
        return ReflectVerdict(
            grounded=True,
            answers_query=False,
            complete=False,
            relevant=True,
            reasoning="On-topic but never states the specific fact asked for.",
        )

    monkeypatch.setattr(webapp_main, "check_groundedness", fake_check_groundedness_llm)

    response = client.post(
        "/api/reflect",
        json={
            "query_text": "what is a corporation",
            "language": "hi",
            "strategy": "native",
            "answer_text": "A corporation is a company.",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["final_status"] == "not_answered"
    assert "does not answer" in data["reason"]


def test_post_reflect_unknown_collection_returns_404(monkeypatch):
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {})
    response = client.post(
        "/api/reflect",
        json={"query_text": "x", "language": "hi", "strategy": "native", "answer_text": "y"},
    )
    assert response.status_code == 404


def test_post_synthesize_returns_answer_when_answerable(monkeypatch):
    from speculative_rag.reflect import SynthesisVerdict

    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored("Acme Corp was founded in 1998.", 0.9, pid="p_founded"),
                _scored("Acme Corp went public in 2004.", 0.85, pid="p_ipo"),
            ]
        )
    )
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    def fake_synthesize_answer_llm(query_text, candidate_texts):
        return SynthesisVerdict(
            answerable=True,
            answer_text="Acme Corp was founded in 1998 and went public in 2004.",
            source_indices=[0, 1],
            reasoning="Combines founding year and IPO year.",
        )

    monkeypatch.setattr(webapp_main, "synthesize_answer", fake_synthesize_answer_llm)

    response = client.post(
        "/api/synthesize",
        json={"query_text": "when was Acme founded and when did it ipo", "language": "hi", "strategy": "native"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answerable"] is True
    assert data["answer_text"] == "Acme Corp was founded in 1998 and went public in 2004."
    assert data["sources"] == [
        {"source_passage_id": "p_founded", "retrieval_score": 0.9},
        {"source_passage_id": "p_ipo", "retrieval_score": 0.85},
    ]
    assert data["synthesis_latency_ms"] >= 0.0


def test_post_synthesize_returns_not_answerable_shape(monkeypatch):
    from speculative_rag.reflect import SynthesisVerdict

    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("unrelated text", 0.9, pid="p1")]))
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    def fake_synthesize_answer_llm(query_text, candidate_texts):
        return SynthesisVerdict(answerable=False, answer_text="", source_indices=[], reasoning="Missing.")

    monkeypatch.setattr(webapp_main, "synthesize_answer", fake_synthesize_answer_llm)

    response = client.post(
        "/api/synthesize",
        json={"query_text": "x", "language": "hi", "strategy": "native"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answerable"] is False
    assert data["answer_text"] == ""
    assert data["sources"] == []


def test_post_synthesize_unknown_collection_returns_404(monkeypatch):
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {})
    response = client.post(
        "/api/synthesize",
        json={"query_text": "x", "language": "hi", "strategy": "native"},
    )
    assert response.status_code == 404


def test_post_rerank_returns_structured_selected_sources(monkeypatch):
    from speculative_rag.reflect import RerankVerdict, SelectedSource

    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    retriever = FakeRetriever(
        RetrievalResult(
            scored_chunks=[
                _scored("Cats are popular household pets.", 0.91, pid="p_rank1"),
                _scored("A corporation is a company or group of people.", 0.80, pid="p_rank2"),
            ]
        )
    )
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    captured = {}

    def fake_rerank_passages_llm(query_text, candidates):
        captured["query_text"] = query_text
        captured["candidate_texts"] = [c.text for c in candidates]
        return RerankVerdict(
            selected_sources=[SelectedSource(source_id=candidates[1].source_id, relevance_score=0.93)],
            answerable=True,
            reasoning="Candidate 1 defines a corporation.",
        )

    monkeypatch.setattr(webapp_main, "rerank_passages", fake_rerank_passages_llm)

    response = client.post(
        "/api/rerank",
        json={"query_text": "what is a corporation", "language": "hi", "strategy": "native"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answerable"] is True
    assert data["selected_sources"] == [
        {"source_id": "p_rank2:native:0", "source_passage_id": "p_rank2", "relevance_score": 0.93}
    ]
    assert data["reasoning"] == "Candidate 1 defines a corporation."
    assert data["rerank_latency_ms"] >= 0.0
    assert captured["query_text"] == "what is a corporation"
    assert captured["candidate_texts"] == [
        "Cats are popular household pets.",
        "A corporation is a company or group of people.",
    ]


def test_post_rerank_returns_empty_selection_when_nothing_relevant(monkeypatch):
    from speculative_rag.reflect import RerankVerdict

    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    retriever = FakeRetriever(
        RetrievalResult(scored_chunks=[_scored("Cats are popular household pets.", 0.91, pid="p_rank1")])
    )
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    def fake_rerank_passages_llm(query_text, candidates):
        return RerankVerdict(selected_sources=[], answerable=False, reasoning="No candidate answers this.")

    monkeypatch.setattr(webapp_main, "rerank_passages", fake_rerank_passages_llm)

    response = client.post(
        "/api/rerank",
        json={"query_text": "what is a corporation", "language": "hi", "strategy": "native"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["selected_sources"] == []
    assert data["answerable"] is False


def test_post_rerank_unknown_collection_returns_404(monkeypatch):
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {})
    response = client.post(
        "/api/rerank",
        json={"query_text": "x", "language": "hi", "strategy": "native"},
    )
    assert response.status_code == 404


def _fake_outcome(**overrides):
    from speculative_rag.answer_orchestrator import AnswerOutcome
    from speculative_rag.answer_status import FinalAnswerStatus
    from speculative_rag.pipeline_schemas import PipelineResult, RagOutcome, StageTiming

    base = dict(
        final_status=FinalAnswerStatus.ANSWERED,
        answer_text="The Manhattan Project successfully produced the atomic bomb, which helped bring an "
        "end to World War II.",
        source_passage_ids=["p_ww2"],
        reason="Directly supported by the evidence.",
        initial_answer_valid=False,
        initial_answerable=False,
        rerank_triggered=True,
        rerank_latency_ms=812.4,
        selected_source_ids=["p_ww2:native:0"],
        reranked_answer_generated=True,
        regenerated_answer_valid=True,
        regenerated_answerable=True,
        synthesis_triggered=False,
        answer_engine="claude_regenerated",
        total_end_to_end_latency_ms=1523.9,
        pipeline_result=PipelineResult(
            outcome=RagOutcome.ANSWERED,
            timings=[StageTiming(stage="embed", elapsed_ms=1.2), StageTiming(stage="retrieve", elapsed_ms=42.0)],
            num_retrieved=5,
            num_evidence_candidates=5,
        ),
    )
    base.update(overrides)
    return AnswerOutcome(**base)


def test_post_answer_returns_fast_provisional_shape_then_polling_returns_final_answer(monkeypatch):
    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("irrelevant for this test", 0.9)]))
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    captured = {}

    def fake_resolve_recovery(query_text, pipeline_result, **kwargs):
        captured["query_text"] = query_text
        captured["pipeline_result"] = pipeline_result
        return _fake_outcome()

    monkeypatch.setattr(webapp_main, "resolve_recovery", fake_resolve_recovery)

    response = client.post(
        "/api/answer",
        json={
            "query_text": "What was the immediate impact of the success of the Manhattan Project?",
            "language": "hi",
            "strategy": "native",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "resolving"
    assert data["request_id"]
    assert "pipeline" in data
    assert captured["query_text"] == "What was the immediate impact of the success of the Manhattan Project?"

    poll = client.get(f"/api/answer/{data['request_id']}")
    assert poll.status_code == 200
    result = poll.json()
    assert result["status"] == "done"
    assert result["final_status"] == "answered"
    assert "atomic bomb" in result["answer"]["text"]
    assert result["answer"]["source_passage_ids"] == ["p_ww2"]
    assert result["answer"]["answer_engine"] == "claude_regenerated"
    assert result["instrumentation"]["rerank_triggered"] is True
    assert result["instrumentation"]["reranked_answer_generated"] is True
    assert result["instrumentation"]["regenerated_answer_valid"] is True
    assert result["instrumentation"]["synthesis_triggered"] is False
    assert result["instrumentation"]["selected_source_ids"] == ["p_ww2:native:0"]
    assert result["instrumentation"]["recovery_latency_ms"] == 1523.9
    assert result["pipeline"]["outcome"] == "answered"
    assert [t["stage"] for t in result["pipeline"]["timings"]] == ["embed", "retrieve"]


def test_post_answer_never_returns_answered_badge_for_unvalidated_result(monkeypatch):
    from speculative_rag.answer_status import FinalAnswerStatus

    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("unrelated text", 0.9)]))
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})

    def fake_resolve_recovery(query_text, pipeline_result, **kwargs):
        return _fake_outcome(
            final_status=FinalAnswerStatus.FALLBACK,
            answer_text=None,
            source_passage_ids=[],
            reason="multi-passage synthesis produced an answer but it failed validation",
            regenerated_answer_valid=False,
            regenerated_answerable=False,
            synthesis_triggered=True,
        )

    monkeypatch.setattr(webapp_main, "resolve_recovery", fake_resolve_recovery)

    response = client.post(
        "/api/answer",
        json={"query_text": "x", "language": "hi", "strategy": "native"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "resolving"

    poll = client.get(f"/api/answer/{data['request_id']}")
    assert poll.status_code == 200
    result = poll.json()
    assert result["final_status"] == "fallback"
    assert result["answer"] is None
    assert result["instrumentation"]["synthesis_triggered"] is True


def test_post_answer_unknown_collection_returns_404(monkeypatch):
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {})
    response = client.post(
        "/api/answer",
        json={"query_text": "x", "language": "hi", "strategy": "native"},
    )
    assert response.status_code == 404


def test_get_answer_unknown_request_id_returns_404():
    response = client.get("/api/answer/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.json()


def test_get_answer_returns_resolving_status_while_pending(monkeypatch):
    monkeypatch.setitem(webapp_main._PENDING_ANSWERS, "req-pending", {"status": "resolving"})
    response = client.get("/api/answer/req-pending")
    assert response.status_code == 200
    assert response.json() == {"status": "resolving"}


def test_pending_answers_evicts_oldest_when_over_cap(monkeypatch):
    from collections import OrderedDict

    monkeypatch.setattr(webapp_main, "_MAX_PENDING_ANSWERS", 2)
    monkeypatch.setattr(webapp_main, "_PENDING_ANSWERS", OrderedDict())
    monkeypatch.setattr(webapp_main, "_EMBEDDERS", {"hi": FakeEmbedder()})
    retriever = FakeRetriever(RetrievalResult(scored_chunks=[_scored("something unrelated", 0.0)]))
    monkeypatch.setattr(webapp_main, "_RETRIEVERS", {"hi_native": retriever})
    monkeypatch.setattr(webapp_main, "resolve_recovery", lambda *a, **k: _fake_outcome())

    request_ids = []
    for _ in range(3):
        response = client.post("/api/answer", json={"query_text": "x", "language": "hi", "strategy": "native"})
        request_ids.append(response.json()["request_id"])

    assert len(webapp_main._PENDING_ANSWERS) == 2
    assert request_ids[0] not in webapp_main._PENDING_ANSWERS
    assert request_ids[1] in webapp_main._PENDING_ANSWERS
    assert request_ids[2] in webapp_main._PENDING_ANSWERS
