from speculative_rag.schemas import Chunk, Passage, QueryRecord


def test_passage_roundtrip():
    p = Passage(passage_id="abc123", text="hello world", language="en", query_type="DESCRIPTION")
    assert p.passage_id == "abc123"
    assert p.model_dump()["text"] == "hello world"


def test_chunk_roundtrip():
    c = Chunk(
        chunk_id="abc123:native:0",
        text="hello world",
        language="en",
        query_type="DESCRIPTION",
        source_passage_id="abc123",
        strategy="native",
    )
    assert c.strategy == "native"


def test_query_record_roundtrip():
    q = QueryRecord(
        query_id=42,
        language="en",
        query_type="NUMERIC",
        query_text="how many?",
        passage_ids=["a", "b"],
        relevant_passage_ids=["b"],
    )
    assert q.relevant_passage_ids == ["b"]
