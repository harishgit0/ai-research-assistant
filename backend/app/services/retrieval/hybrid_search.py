from app.services.retrieval.bm25_search import bm25_search
from app.services.retrieval.semantic_search import semantic_search


def hybrid_search(
    query: str,
    top_k: int = 5,
    retrieval_k: int = 20,
    rrf_k: int = 60,
    document_id: int | None = None,
) -> list[dict]:
    """Combine semantic and BM25 retrieval using Reciprocal Rank Fusion."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if retrieval_k <= 0:
        raise ValueError("retrieval_k must be greater than 0.")

    semantic_results = semantic_search(
        query,
        top_k=retrieval_k,
        document_id=document_id,
    )
    bm25_results = bm25_search(
        query,
        top_k=retrieval_k,
        document_id=document_id,
    )

    ranked_results = {}

    for rank, result in enumerate(semantic_results, start=1):
        chunk_id = result["chunk_id"]
        if chunk_id not in ranked_results:
            ranked_results[chunk_id] = {
                "chunk_id": result["chunk_id"],
                "document_id": result["document_id"],
                "filename": result["filename"],
                "chunk_index": result["chunk_index"],
                "page_number": result["page_number"],
                "text": result["text"],
                "semantic_rank": rank,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }
        ranked_results[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]
        if chunk_id not in ranked_results:
            ranked_results[chunk_id] = {
                "chunk_id": result["chunk_id"],
                "document_id": result["document_id"],
                "filename": result["filename"],
                "chunk_index": result["chunk_index"],
                "page_number": result["page_number"],
                "text": result["text"],
                "semantic_rank": None,
                "bm25_rank": rank,
                "rrf_score": 0.0,
            }
        ranked_results[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)
        ranked_results[chunk_id]["bm25_rank"] = rank

    results = sorted(
        ranked_results.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )
    return results[:top_k]
