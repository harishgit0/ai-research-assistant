from rank_bm25 import BM25Okapi

from app.db.connection import get_connection


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 search."""
    return text.lower().split()


def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Search document chunks using BM25 lexical matching.

    Args:
        query: User's search question.
        top_k: Number of results to return.

    Returns:
        A list of the highest-scoring chunks.
    """

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
                JOIN documents d
                    ON c.document_id = d.id
                WHERE c.text IS NOT NULL
                ORDER BY c.id;
                """
            )

            rows = cursor.fetchall()

    finally:
        connection.close()

    if not rows:
        return []

    tokenized_corpus = [
        _tokenize(row[5])
        for row in rows
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices[:top_k]:
        row = rows[index]

        results.append({
            "chunk_id": row[0],
            "document_id": row[1],
            "filename": row[2],
            "chunk_index": row[3],
            "page_number": row[4],
            "text": row[5],
            "bm25_score": float(scores[index]),
        })

    return results