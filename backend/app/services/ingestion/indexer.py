from pathlib import Path

from app.db.connection import get_connection
from app.services.embedding import EmbeddingService
from app.services.ingestion.chunker import chunk_text
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.pdf import extract_text_from_pdf


def index_pdf(file_path: str) -> int:
    """
    Extract, clean, chunk, embed, and store a PDF in PostgreSQL.

    Returns the number of chunks stored.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = extract_text_from_pdf(str(pdf_path))

    chunks = []

    document_chunk_index = 0

    for page in pages:
        cleaned_text = clean_text(page["text"])
        page_chunks = chunk_text(cleaned_text)

        for chunk in page_chunks:
            chunks.append({
                "chunk_index": document_chunk_index,
                "page_number": page["page_number"],
                "text": chunk["text"],
                "token_count": chunk["token_count"],
            })

            document_chunk_index += 1

    if not chunks:
        raise ValueError("No chunks were generated from the PDF.")

    embedding_service = EmbeddingService()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.embed_texts(texts)

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            # Remove any existing copy of this document.
            # chunks are deleted automatically because of ON DELETE CASCADE.
            cursor.execute(
                """
                DELETE FROM documents
                WHERE filename = %s;
                """,
                (pdf_path.name,),
            )

            cursor.execute(
                """
                INSERT INTO documents (filename, file_path)
                VALUES (%s, %s)
                RETURNING id;
                """,
                (pdf_path.name, str(pdf_path)),
            )

            document_id = cursor.fetchone()[0]

            for chunk, embedding in zip(chunks, embeddings):
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        chunk_index,
                        page_number,
                        text,
                        token_count,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (
                        document_id,
                        chunk["chunk_index"],
                        chunk["page_number"],
                        chunk["text"],
                        chunk["token_count"],
                        embedding.tolist(),
                    ),
                )
            connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    print(f"Indexed document: {pdf_path.name}")
    print(f"Document ID: {document_id}")
    print(f"Chunks stored: {len(chunks)}")

    return len(chunks)
