from app.services.retrieval.bm25_search import bm25_search


def test_bm25_search():
    query = "retrieval augmented generation"

    results = bm25_search(query, top_k=5)

    assert len(results) == 5

    for result in results:
        print("\n--- BM25 Result ---")
        print("Chunk ID:", result["chunk_id"])
        print("Document:", result["filename"])
        print("Page:", result["page_number"])
        print("BM25 Score:", result["bm25_score"])
        print("Text:", result["text"][:500])

        assert "text" in result
        assert "bm25_score" in result
        assert "filename" in result
        assert result["filename"] == "RAG-Survey.pdf"
        assert result["bm25_score"] >= 0

    top_result = results[0]

    assert any(
        term in top_result["text"].lower()
        for term in [
            "retrieval",
            "generation",
            "retrieval-augmented",
        ]
    )	
