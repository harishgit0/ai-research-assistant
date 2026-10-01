from app.services.retrieval.retrieval_pipeline import retrieve


def test_retrieval_pipeline():
    query = "What is retrieval augmented generation?"

    results = retrieve(
        query,
        top_k=5,
        candidate_k=20,
    )

    assert len(results) == 5

    for result in results:
        print("\n--- Final Retrieval Result ---")
        print("Chunk ID:", result["chunk_id"])
        print("Document:", result["filename"])
        print("Page:", result["page_number"])
        print("Semantic Rank:", result["semantic_rank"])
        print("BM25 Rank:", result["bm25_rank"])
        print("RRF Score:", result["rrf_score"])
        print("Reranker Score:", result["reranker_score"])
        print("Text:", result["text"][:500])

        assert result["filename"] == "RAG-Survey.pdf"
        assert "rrf_score" in result
        assert "reranker_score" in result

    assert len({
        result["chunk_id"]
        for result in results
    }) == 5
