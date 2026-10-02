from app.services.retrieval.retrieval_pipeline import retrieve


def test_retrieval_pipeline():
    results = retrieve(
        "What is retrieval augmented generation?",
        top_k=5,
        candidate_k=20,
    )

    assert results
    assert len(results) <= 5

    for result in results:
        assert result["filename"]
        assert result["text"]
        assert "rrf_score" in result
        assert "reranker_score" in result

    assert len({result["chunk_id"] for result in results}) == len(results)
