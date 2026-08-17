from qdrant_client import QdrantClient

from scripts.run_recall_eval import run_comparison
from speculative_rag.config import LanguageSource, Settings
from speculative_rag.sparse_embedding import SparseVectorData


class FakeEmbedder:
    """Deterministic hash-based sparse embedding — no model download needed."""

    def _vec(self, text: str) -> SparseVectorData:
        tokens = text.lower().split()
        return SparseVectorData(indices=[hash(t) % 1000 for t in tokens], values=[1.0] * len(tokens))

    def embed_query(self, text: str) -> SparseVectorData:
        return self._vec(text)

    def embed_passages(self, texts: list[str]) -> list[SparseVectorData]:
        return [self._vec(t) for t in texts]


def test_run_comparison_produces_rows_for_every_strategy_and_k(tmp_path):
    import numpy as np
    import pandas as pd

    rows = [
        {
            "query_id": 1,
            "query_type": "DESCRIPTION",
            "query": "q1",
            "passages": {
                "Translated_passages": np.array(["gold passage text", "distractor text"], dtype=object),
                "English_passages": np.array(["gold passage text", "distractor text"], dtype=object),
                "is_selected": np.array([1, 0], dtype=np.int32),
            },
        }
    ]
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "fake.parquet"
    df.to_parquet(parquet_path)

    sources = [LanguageSource(language="hi", parquet_path=parquet_path, text_field="Translated_passages")]
    settings = Settings(sample_size=10, qdrant_path=tmp_path / "qdrant")
    client = QdrantClient(path=str(settings.qdrant_path))

    results = run_comparison(sources, lambda language: FakeEmbedder(), client, settings)

    strategies = {r["strategy"] for r in results}
    assert strategies == {"native", "sliding_window", "sentence", "metadata"}
    assert all(r["num_queries"] == 1 for r in results)
