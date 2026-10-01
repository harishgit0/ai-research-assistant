from unittest.mock import Mock, patch

import pytest

from app.services.research_assistant_service import (
    ResearchAssistantService,
)


@pytest.fixture
def retrieval_results():
    return [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "PUBLIC MONEY.pdf",
            "chunk_index": 0,
            "page_number": 1,
            "text": "The project has an estimated cost of Rs. 10 lakh.",
            "rrf_score": 0.032,
        },
        {
            "chunk_id": 2,
            "document_id": 1,
            "filename": "PUBLIC MONEY.pdf",
            "chunk_index": 1,
            "page_number": 2,
            "text": "The project was completed in March 2026.",
            "rrf_score": 0.031,
        },
    ]


def test_research_assistant_runs_retrieval_reranking_and_generation(
    retrieval_results,
):
    rag_service = Mock()
    rag_service.generate_answer.return_value = (
        "The project cost is Rs. 10 lakh."
    )

    reranker = Mock()
    reranker.rerank.return_value = retrieval_results

    service = ResearchAssistantService(
        rag_service=rag_service,
        reranker=reranker,
    )

    with patch(
        "app.services.research_assistant_service.hybrid_search"
    ) as mock_hybrid_search:
        mock_hybrid_search.return_value = retrieval_results

        result = service.answer(
            question="What is the project cost?",
        )

    mock_hybrid_search.assert_called_once_with(
        query="What is the project cost?",
        retrieval_k=20,
        top_k=20,
    )

    reranker.rerank.assert_called_once_with(
        query="What is the project cost?",
        results=retrieval_results,
        top_k=5,
    )

    rag_service.generate_answer.assert_called_once_with(
        question="What is the project cost?",
        retrieval_results=retrieval_results,
    )

    assert result["answer"] == "The project cost is Rs. 10 lakh."
    assert result["sources"] == retrieval_results


def test_research_assistant_passes_custom_parameters(
    retrieval_results,
):
    rag_service = Mock()
    rag_service.generate_answer.return_value = "Answer"

    reranker = Mock()
    reranker.rerank.return_value = retrieval_results

    service = ResearchAssistantService(
        rag_service=rag_service,
        reranker=reranker,
    )

    with patch(
        "app.services.research_assistant_service.hybrid_search"
    ) as mock_hybrid_search:
        mock_hybrid_search.return_value = retrieval_results

        service.answer(
            question="What is the project cost?",
            retrieval_k=10,
            top_k=3,
        )

    mock_hybrid_search.assert_called_once_with(
        query="What is the project cost?",
        retrieval_k=10,
        top_k=10,
    )

    reranker.rerank.assert_called_once_with(
        query="What is the project cost?",
        results=retrieval_results,
        top_k=3,
    )


def test_research_assistant_rejects_empty_question():
    rag_service = Mock()
    reranker = Mock()

    service = ResearchAssistantService(
        rag_service=rag_service,
        reranker=reranker,
    )

    with pytest.raises(
        ValueError,
        match="Question must not be empty",
    ):
        service.answer("   ")


def test_research_assistant_rejects_invalid_retrieval_k():
    rag_service = Mock()
    reranker = Mock()

    service = ResearchAssistantService(
        rag_service=rag_service,
        reranker=reranker,
    )

    with pytest.raises(
        ValueError,
        match="retrieval_k must be greater than 0",
    ):
        service.answer(
            question="What is the project cost?",
            retrieval_k=0,
        )


def test_research_assistant_rejects_invalid_top_k():
    rag_service = Mock()
    reranker = Mock()

    service = ResearchAssistantService(
        rag_service=rag_service,
        reranker=reranker,
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        service.answer(
            question="What is the project cost?",
            top_k=0,
        )