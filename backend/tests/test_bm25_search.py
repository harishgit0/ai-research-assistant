from app.services.retrieval.bm25_search import bm25_search


def test_bm25_search():
    results = bm25_search("retrieval augmented generation", top_k=5)

    assert len(results) <= 5
    assert results

    for result in results:
        assert result["text"]
        assert result["filename"]
        assert result["bm25_score"] >= 0
