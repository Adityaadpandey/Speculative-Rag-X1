from sentence_transformers import SentenceTransformer

from speculative_rag.config import SETTINGS


class Embedder:
    def __init__(self, model_name: str = SETTINGS.embedding_model):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()
