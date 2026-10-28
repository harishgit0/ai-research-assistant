from pathlib import Path

from pgvector import Vector

from app.db.connection import get_connection
from app.services.embedding import EmbeddingService
from app.services.ingestion.chunker import chunk_text
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.pdf import extract_text_from_pdf


def index_pdf(file_path: str) -> int:
    """Extract, clean, chunk, embed, and transactionally store a PDF."""
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    chunks = []
    next_index = 0
    for page in extract_text_from_pdf(str(pdf_path)):
        cleaned = clean_text(page["text"], page["page_number"])
        for chunk in chunk_text(cleaned):
            chunks.append({
                "chunk_index": next_index,
                "page_number": page["page_number"],
                "text": chunk["text"],
                "token_count": chunk["token_count"],
            })
            next_index += 1

    if not chunks:
        raise ValueError(
            "No extractable text was found. Scanned PDFs need OCR before indexing."
        )

    embeddings = EmbeddingService().embed_texts([item["text"] for item in chunks])
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Replacement is atomic: an insertion failure rolls back the deletion.
            cur.execute("DELETE FROM documents WHERE filename = %s;", (pdf_path.name,))
            cur.execute(
                "INSERT INTO documents (filename, file_path) VALUES (%s, %s) RETURNING id;",
                (pdf_path.name, str(pdf_path.resolve())),
            )
            document_id = cur.fetchone()[0]
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute("""
                    INSERT INTO chunks (
                        document_id, chunk_index, page_number, text,
                        token_count, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s::vector);
                """, (
                    document_id, chunk["chunk_index"], chunk["page_number"],
                    chunk["text"], chunk["token_count"],
                    str(Vector(embedding.tolist())),
                ))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return len(chunks)
