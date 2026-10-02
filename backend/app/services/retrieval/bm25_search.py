from rank_bm25 import BM25Okapi

from app.db.connection import get_connection


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 search."""
    return text.lower().split()


def bm25_search(
    query: str,
    top_k: int = 5,
    document_id: int | None = None,
) -> list[dict]:
    """Search chunks lexically, optionally within one document."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    c.page_number,
                    c.text
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.text IS NOT NULL
                  AND (%s IS NULL OR c.document_id = %s)
                ORDER BY c.id;
                """,
                (document_id, document_id),
            )
            rows = cursor.fetchall()
    finally:
        connection.close()

    if not rows:
        return []

    tokenized_corpus = [_tokenize(row[5]) for row in rows]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    return [
        {
            "chunk_id": rows[index][0],
            "document_id": rows[index][1],
            "filename": rows[index][2],
            "chunk_index": rows[index][3],
            "page_number": rows[index][4],
            "text": rows[index][5],
            "bm25_score": float(scores[index]),
        }
        for index in ranked_indices[:top_k]
    ]
