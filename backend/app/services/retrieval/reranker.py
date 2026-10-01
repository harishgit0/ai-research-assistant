from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker:
    """Cross-encoder based document reranker."""

    def __init__(self):
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank retrieval candidates using a cross-encoder.

        Args:
            query: User's search query.
            results: Candidate chunks from retrieval.
            top_k: Number of final results.

        Returns:
            Reranked chunks with reranker scores.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not results:
            return []

        pairs = [
            (query, result["text"])
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):
            reranked_result = {
                **result,
                "reranker_score": float(score),
            }

            reranked_results.append(reranked_result)

        reranked_results.sort(
            key=lambda result: result["reranker_score"],
            reverse=True,
        )

        return reranked_results[:top_k]
