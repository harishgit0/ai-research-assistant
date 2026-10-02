from app.services.generation.rag_service import RAGService
from app.services.retrieval.hybrid_search import hybrid_search
from app.services.retrieval.reranker import Reranker


class ResearchAssistantService:
    """Orchestrate retrieval, reranking, and answer generation."""

    def __init__(
        self,
        rag_service: RAGService,
        reranker: Reranker | None = None,
    ):
        self.rag_service = rag_service
        self.reranker = reranker or Reranker()

    def answer(
        self,
        question: str,
        retrieval_k: int = 20,
        top_k: int = 5,
        document_id: int | None = None,
    ) -> dict:
        """Answer using all documents or only the selected document."""

        if not question or not question.strip():
            raise ValueError("Question must not be empty.")
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be greater than 0.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        retrieval_kwargs = {
            "query": question,
            "retrieval_k": retrieval_k,
            "top_k": retrieval_k,
        }

        # Preserve the original call signature for global search.
        # Pass document_id only when the caller explicitly scopes the query.
        if document_id is not None:
            retrieval_kwargs["document_id"] = document_id

        retrieval_results = hybrid_search(**retrieval_kwargs)

        if not retrieval_results:
            raise ValueError("No searchable chunks found for the selected document.")

        reranked_results = self.reranker.rerank(
            query=question,
            results=retrieval_results,
            top_k=top_k,
        )
        answer = self.rag_service.generate_answer(
            question=question,
            retrieval_results=reranked_results,
        )
        return {"answer": answer, "sources": reranked_results}
