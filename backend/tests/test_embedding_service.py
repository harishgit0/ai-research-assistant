from app.services.embedding import (
    EmbeddingService,
    EMBEDDING_DIMENSION,
)


def test_embedding_dimension():
    service = EmbeddingService()

    embedding = service.embed_text(
        "A government project exceeded its expected cost."
    )

    assert embedding.shape == (EMBEDDING_DIMENSION,)


def test_multiple_embeddings():
    service = EmbeddingService()

    embeddings = service.embed_texts(
        [
            "Government project expenditure",
            "Construction cost exceeded the estimate",
        ]
    )

    assert embeddings.shape == (2, EMBEDDING_DIMENSION)
