from qdrant_client import QdrantClient

from scripts.run_recall_eval import run_comparison
from speculative_rag.config import LanguageSource, Settings


class FakeEmbedder:
    """Deterministic hash-based embedding — no model download needed."""

    dimension = 8

    def _vec(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha1(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dimension]]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
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

    results = run_comparison(sources, FakeEmbedder(), client, settings)

    strategies = {r["strategy"] for r in results}
    assert strategies == {"native", "sliding_window", "sentence", "metadata"}
    assert all(r["num_queries"] == 1 for r in results)
