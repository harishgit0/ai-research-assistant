from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker:
    """Cross-encoder reranker; the model loads only on first use."""

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = CrossEncoder(MODEL_NAME)
        return self._model

    def rerank(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not results:
            return []
        scores = self.model.predict([(query, result["text"]) for result in results])
        ranked = [
            {**result, "reranker_score": float(score)}
            for result, score in zip(results, scores)
        ]
        ranked.sort(key=lambda result: result["reranker_score"], reverse=True)
        return ranked[:top_k]
