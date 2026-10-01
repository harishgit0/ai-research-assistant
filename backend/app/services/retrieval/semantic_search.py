from app.db.connection import get_connection
from app.services.embedding import EmbeddingService


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Search stored document chunks using semantic similarity.

    Args:
        query: User's search question.
        top_k: Number of results to return.

    Returns:
        A list of the most semantically similar chunks.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    embedding_service = EmbeddingService()

    query_embedding = embedding_service.embed_text(query)

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
                    c.text,
                    1 - (c.embedding <=> %s::vector) AS similarity
                FROM chunks c
                JOIN documents d
                    ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s;
                """,
                (
                    query_embedding.tolist(),
                    query_embedding.tolist(),
                    top_k,
                ),
            )

            rows = cursor.fetchall()

        results = []

        for row in rows:
            results.append({
                "chunk_id": row[0],
                "document_id": row[1],
                "filename": row[2],
                "chunk_index": row[3],
                "page_number": row[4],
                "text": row[5],
                "similarity": float(row[6]),
            })

        return results

    finally:
        connection.close()