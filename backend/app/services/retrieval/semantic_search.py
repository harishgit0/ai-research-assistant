from pgvector import Vector

from app.db.connection import get_connection
from app.services.embedding import EmbeddingService


def semantic_search(
    query: str,
    top_k: int = 5,
    document_id: int | None = None,
) -> list[dict]:
    """Search stored document chunks by cosine similarity."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    embedding = EmbeddingService().embed_text(query)
    vector = str(Vector(embedding.tolist()))
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.document_id, d.filename, c.chunk_index,
                       c.page_number, c.text,
                       1 - (c.embedding <=> %s::vector) AS similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                  AND (%s::integer IS NULL OR c.document_id = %s)
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s;
            """, (vector, document_id, document_id, vector, top_k))
            rows = cur.fetchall()
        return [{
            "chunk_id": row[0], "document_id": row[1], "filename": row[2],
            "chunk_index": row[3], "page_number": row[4], "text": row[5],
            "similarity": float(row[6]),
        } for row in rows]
    finally:
        conn.close()
