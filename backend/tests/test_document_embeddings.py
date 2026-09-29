from app.services.embedding import EmbeddingService
from app.services.ingestion.pdf import extract_text_from_pdf
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.chunker import chunk_text


PDF_PATH = "data/uploads/PUBLIC MONEY.pdf"


def test_public_money_embeddings():
    pages = extract_text_from_pdf(PDF_PATH)

    chunks = []

    for page in pages:
        cleaned_text = clean_text(page["text"])

        page_chunks = chunk_text(cleaned_text)

        for chunk in page_chunks:
            chunks.append({
                "page_number": page["page_number"],
                **chunk,
            })

    assert len(chunks) > 0

    texts = [chunk["text"] for chunk in chunks]

    service = EmbeddingService()
    embeddings = service.embed_texts(texts)

    print("\nNumber of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("Expected dimensions:", 384)

    assert embeddings.shape == (len(chunks), 384)

    first_vector_norm = sum(
        x * x for x in embeddings[0]
    ) ** 0.5

    print("First vector norm:", first_vector_norm)

    assert abs(first_vector_norm - 1.0) < 0.001
