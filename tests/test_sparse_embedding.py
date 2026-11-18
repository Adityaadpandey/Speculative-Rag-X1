import pytest

from speculative_rag.sparse_embedding import SparseEmbedder


@pytest.mark.slow
def test_embed_query_and_passages_share_hash_space():
    embedder = SparseEmbedder("en")
    query_vec = embedder.embed_query("what is a corporation?")
    passage_vecs = embedder.embed_passages(["a corporation is a company", "unrelated text about weather"])

    assert query_vec.indices
    assert set(query_vec.indices) & set(passage_vecs[0].indices)


@pytest.mark.slow
def test_relevant_passage_overlaps_query_more_than_irrelevant():
    embedder = SparseEmbedder("en")
    query_vec = embedder.embed_query("what is a corporation?")
    relevant_vec = embedder.embed_passages(["A corporation is a company recognized in law."])[0]
    irrelevant_vec = embedder.embed_passages(["The weather today is sunny with light wind."])[0]

    relevant_overlap = len(set(query_vec.indices) & set(relevant_vec.indices))
    irrelevant_overlap = len(set(query_vec.indices) & set(irrelevant_vec.indices))
    assert relevant_overlap > irrelevant_overlap


@pytest.mark.slow
def test_hindi_falls_back_to_unstemmed_tokenization_without_erroring():
    embedder = SparseEmbedder("hi")
    query_vec = embedder.embed_query("भारत की राजधानी क्या है")
    assert query_vec.indices


def test_hindi_words_survive_tokenization_intact():
    # Regression: fastembed's stock text cleanup dropped Devanagari vowel
    # signs, so "भारत" was indexed as the fragments "भ" + "रत".
    from speculative_rag.sparse_embedding import SparseEmbedder, _indic_safe_words

    assert _indic_safe_words("भारत की राजधानी?") == ["भारत", "की", "राजधानी"]
    embedder = SparseEmbedder("hi")
    whole = set(embedder.embed_query("भारत").indices)
    assert len(whole) == 1
    assert not whole & set(embedder.embed_query("रत").indices)
    # Passage and query sides must hash the same word to the same term.
    assert whole <= set(embedder.embed_passages(["भारत दक्षिण एशिया में है"])[0].indices)


def test_english_tokenization_is_unchanged():
    from fastembed import SparseTextEmbedding

    from speculative_rag.sparse_embedding import SparseEmbedder

    stock = SparseTextEmbedding("Qdrant/bm25", language="english")
    text = "Where is India located? It's in South Asia."
    ours = SparseEmbedder("en").embed_passages([text])[0]
    theirs = next(iter(stock.embed([text])))
    assert ours.indices == theirs.indices.tolist()
