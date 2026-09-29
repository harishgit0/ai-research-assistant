from app.db.connection import get_connection
from app.services.ingestion.indexer import index_pdf


PDF_PATH = "backend/data/uploads/RAG-Survey.pdf"
PDF_FILENAME = "RAG-Survey.pdf"


def test_index_rag_survey_pdf():
    chunks_stored = index_pdf(PDF_PATH)

    assert chunks_stored > 0

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM chunks c
                JOIN documents d
                    ON c.document_id = d.id
                WHERE d.filename = %s;
                """,
                (PDF_FILENAME,),
            )

            stored_count = cursor.fetchone()[0]

            assert stored_count == chunks_stored

    finally:
        connection.close()
