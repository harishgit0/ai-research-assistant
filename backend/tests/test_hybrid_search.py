from app.services.retrieval.hybrid_search import hybrid_search


def test_hybrid_search():
    results = hybrid_search(
        "What is retrieval augmented generation?",
        top_k=5,
        retrieval_k=20,
    )

    assert results
    assert len(results) <= 5

    for result in results:
        assert result["text"]
        assert result["filename"]
        assert result["rrf_score"] > 0
        assert "semantic_rank" in result
        assert "bm25_rank" in result

    assert len({result["chunk_id"] for result in results}) == len(results)
