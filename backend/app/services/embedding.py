from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384


class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_text(self, text: str):
        return self.model.encode(
            text,
            normalize_embeddings=True,
        )

    def embed_texts(self, texts: list[str]):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )
