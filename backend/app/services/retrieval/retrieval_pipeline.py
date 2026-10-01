from app.services.retrieval.hybrid_search import hybrid_search
from app.services.retrieval.reranker import Reranker


def retrieve(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[dict]:
    """
    Run the complete retrieval pipeline.

    Pipeline:
        Semantic + BM25
            ↓
        RRF fusion
            ↓
        Cross-encoder reranking
            ↓
        Final top-K results
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0.")

    candidates = hybrid_search(
        query,
        top_k=candidate_k,
        retrieval_k=candidate_k,
    )

    reranker = Reranker()

    return reranker.rerank(
        query,
        candidates,
        top_k=top_k,
    )
