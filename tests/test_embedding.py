import pytest

from speculative_rag.embedding import Embedder


@pytest.mark.slow
def test_embed_query_and_passages_same_dimension():
    embedder = Embedder()
    query_vec = embedder.embed_query("what is a corporation?")
    passage_vecs = embedder.embed_passages(["a corporation is a company", "unrelated text about weather"])

    assert len(query_vec) == embedder.dimension
    assert all(len(v) == embedder.dimension for v in passage_vecs)


@pytest.mark.slow
def test_relevant_passage_scores_higher_than_irrelevant():
    import numpy as np

    embedder = Embedder()
    query_vec = np.array(embedder.embed_query("what is a corporation?"))
    relevant_vec = np.array(embedder.embed_passages(["A corporation is a company recognized in law."])[0])
    irrelevant_vec = np.array(embedder.embed_passages(["The weather today is sunny with light wind."])[0])

    assert float(query_vec @ relevant_vec) > float(query_vec @ irrelevant_vec)
