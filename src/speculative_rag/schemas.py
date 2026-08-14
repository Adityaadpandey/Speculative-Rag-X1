from pydantic import BaseModel


class Passage(BaseModel):
    passage_id: str
    text: str
    language: str
    query_type: str


class Chunk(BaseModel):
    chunk_id: str
    text: str
    language: str
    query_type: str
    source_passage_id: str
    strategy: str


class QueryRecord(BaseModel):
    query_id: int
    language: str
    query_type: str
    query_text: str
    passage_ids: list[str]
    relevant_passage_ids: list[str]
