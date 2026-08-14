"""Chunking strategies operate within a single passage, never across passages
in the same query's candidate set -- MS MARCO candidate passages come from
unrelated source documents, so concatenating them would fabricate adjacency
that isn't real. Most passages are short enough that sliding_window and
sentence_aware collapse to the same single chunk as native; that's an
expected outcome we see in the Recall@K numbers, not a bug.
"""

import re

from speculative_rag.schemas import Chunk, Passage

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text]


def chunk_native(passage: Passage) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{passage.passage_id}:native:0",
            text=passage.text,
            language=passage.language,
            query_type=passage.query_type,
            source_passage_id=passage.passage_id,
            strategy="native",
        )
    ]


def chunk_sliding_window(passage: Passage, window_words: int = 40, overlap_words: int = 10) -> list[Chunk]:
    words = passage.text.split()
    if len(words) <= window_words:
        return [
            Chunk(
                chunk_id=f"{passage.passage_id}:sliding_window:0",
                text=passage.text,
                language=passage.language,
                query_type=passage.query_type,
                source_passage_id=passage.passage_id,
                strategy="sliding_window",
            )
        ]

    step = window_words - overlap_words
    chunks = []
    idx = 0
    start = 0
    while start < len(words):
        window = words[start : start + window_words]
        chunks.append(
            Chunk(
                chunk_id=f"{passage.passage_id}:sliding_window:{idx}",
                text=" ".join(window),
                language=passage.language,
                query_type=passage.query_type,
                source_passage_id=passage.passage_id,
                strategy="sliding_window",
            )
        )
        idx += 1
        start += step
    return chunks


def chunk_sentence_aware(passage: Passage, max_words: int = 40, overlap_sentences: int = 1) -> list[Chunk]:
    sentences = split_sentences(passage.text)
    if len(sentences) <= 1:
        return [
            Chunk(
                chunk_id=f"{passage.passage_id}:sentence:0",
                text=passage.text,
                language=passage.language,
                query_type=passage.query_type,
                source_passage_id=passage.passage_id,
                strategy="sentence",
            )
        ]

    chunks = []
    idx = 0
    start = 0
    while start < len(sentences):
        group: list[str] = []
        word_count = 0
        i = start
        while i < len(sentences) and (word_count == 0 or word_count + len(sentences[i].split()) <= max_words):
            group.append(sentences[i])
            word_count += len(sentences[i].split())
            i += 1
        chunks.append(
            Chunk(
                chunk_id=f"{passage.passage_id}:sentence:{idx}",
                text=" ".join(group),
                language=passage.language,
                query_type=passage.query_type,
                source_passage_id=passage.passage_id,
                strategy="sentence",
            )
        )
        idx += 1
        if i >= len(sentences):
            break
        start = max(i - overlap_sentences, start + 1)
    return chunks


def chunk_metadata_aware(passage: Passage) -> list[Chunk]:
    native = chunk_native(passage)[0]
    return [native.model_copy(update={"chunk_id": f"{passage.passage_id}:metadata:0", "strategy": "metadata"})]


CHUNKERS = {
    "native": chunk_native,
    "sliding_window": chunk_sliding_window,
    "sentence": chunk_sentence_aware,
    "metadata": chunk_metadata_aware,
}
