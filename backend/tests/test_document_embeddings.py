from pathlib import Path

import pytest

from app.services.embedding import EmbeddingService
from app.services.ingestion.pdf import extract_text_from_pdf
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.chunker import chunk_text


PDF_PATH = Path(__file__).resolve().parents[1] / "data" / "uploads" / "PUBLIC MONEY.pdf"


def test_public_money_embeddings():
    if not PDF_PATH.exists():
        pytest.skip(f"Optional local PDF fixture not found: {PDF_PATH}")

    pages = extract_text_from_pdf(str(PDF_PATH))

    chunks = []
    for page in pages:
        cleaned_text = clean_text(page["text"])
        for chunk in chunk_text(cleaned_text):
            chunks.append({"page_number": page["page_number"], **chunk})

    assert chunks

    embeddings = EmbeddingService().embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    assert embeddings.shape == (len(chunks), 384)
    first_vector_norm = sum(x * x for x in embeddings[0]) ** 0.5
    assert abs(first_vector_norm - 1.0) < 0.001
