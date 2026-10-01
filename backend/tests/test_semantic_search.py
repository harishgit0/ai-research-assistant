from app.services.retrieval.semantic_search import semantic_search


def test_semantic_search():
    query = "What is retrieval augmented generation?"

    results = semantic_search(query, top_k=5)

    assert len(results) == 5

    for result in results:
        print("\n--- Result ---")
        print("Chunk ID:", result["chunk_id"])
        print("Document:", result["filename"])
        print("Page:", result["page_number"])
        print("Similarity:", result["similarity"])
        print("Text:", result["text"][:500])

        assert "text" in result
        assert "similarity" in result
        assert "filename" in result
        assert result["filename"] == "RAG-Survey.pdf"
        assert 0 <= result["similarity"] <= 1

    top_result = results[0]

    assert any(
        term in top_result["text"].lower()
        for term in [
            "retrieval",
            "generation",
            "knowledge base",
            "retrieval-augmented",
        ]
    )