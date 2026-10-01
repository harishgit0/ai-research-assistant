from app.services.retrieval.hybrid_search import hybrid_search


def test_hybrid_search():
    query = "What is retrieval augmented generation?"

    results = hybrid_search(
        query,
        top_k=5,
        retrieval_k=20,
    )

    assert len(results) == 5

    for result in results:
        print("\n--- Hybrid Result ---")
        print("Chunk ID:", result["chunk_id"])
        print("Document:", result["filename"])
        print("Page:", result["page_number"])
        print("Semantic Rank:", result["semantic_rank"])
        print("BM25 Rank:", result["bm25_rank"])
        print("RRF Score:", result["rrf_score"])
        print("Text:", result["text"][:500])

        assert "text" in result
        assert "rrf_score" in result
        assert "semantic_rank" in result
        assert "bm25_rank" in result
        assert result["filename"] == "RAG-Survey.pdf"
        assert result["rrf_score"] > 0

    assert len({
        result["chunk_id"]
        for result in results
    }) == 5
