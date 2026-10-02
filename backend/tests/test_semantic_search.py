from app.services.retrieval.semantic_search import semantic_search


def test_semantic_search():
    results = semantic_search(
        "What is retrieval augmented generation?",
        top_k=5,
    )

    assert results
    assert len(results) <= 5

    for result in results:
        assert result["text"]
        assert result["similarity"] is not None
        assert result["filename"]
        assert 0 <= result["similarity"] <= 1
