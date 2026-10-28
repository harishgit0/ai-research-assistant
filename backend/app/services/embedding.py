from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the embedding model once per process."""
    return SentenceTransformer(MODEL_NAME)


class EmbeddingService:
    def __init__(self):
        self.model = _load_model()

    def embed_text(self, text: str):
        if not text or not text.strip():
            raise ValueError("Text to embed must not be empty.")
        return self.model.encode(text, normalize_embeddings=True)

    def embed_texts(self, texts: list[str]):
        if not texts:
            raise ValueError("At least one text is required for embedding.")
        if any(not text or not text.strip() for text in texts):
            raise ValueError("Texts to embed must not contain empty values.")
        return self.model.encode(texts, normalize_embeddings=True)
