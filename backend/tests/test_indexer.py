from pathlib import Path

import pytest

from app.db.connection import get_connection
from app.services.ingestion.indexer import index_pdf


PDF_PATH = Path(__file__).resolve().parents[1] / "data" / "uploads" / "RAG-Survey.pdf"
PDF_FILENAME = "RAG-Survey.pdf"


@pytest.mark.skipif(
    not PDF_PATH.exists(),
    reason="Optional local RAG-Survey.pdf fixture is not present.",
)
def test_index_rag_survey_pdf():
    chunks_stored = index_pdf(str(PDF_PATH))

    assert chunks_stored > 0

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.filename = %s;
                """,
                (PDF_FILENAME,),
            )
            assert cursor.fetchone()[0] == chunks_stored
    finally:
        connection.close()
