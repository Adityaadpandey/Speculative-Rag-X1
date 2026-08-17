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
