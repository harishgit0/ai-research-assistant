from app.services.retrieval.reranker import Reranker


def test_reranker():
    query = "What is retrieval augmented generation?"

    candidates = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "test.pdf",
            "chunk_index": 0,
            "page_number": 1,
            "text": (
                "Retrieval augmented generation combines "
                "retrieval with language generation."
            ),
        },
        {
            "chunk_id": 2,
            "document_id": 1,
            "filename": "test.pdf",
            "chunk_index": 1,
            "page_number": 2,
            "text": (
                "The weather forecast predicts heavy rainfall "
                "throughout the weekend."
            ),
        },
    ]

    reranker = Reranker()

    results = reranker.rerank(
        query,
        candidates,
        top_k=2,
    )

    assert len(results) == 2

    for result in results:
        print("\n--- Reranked Result ---")
        print("Chunk ID:", result["chunk_id"])
        print("Score:", result["reranker_score"])
        print("Text:", result["text"])

        assert "reranker_score" in result
        assert isinstance(result["reranker_score"], float)

    assert results[0]["chunk_id"] == 1
