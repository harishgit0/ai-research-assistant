import pytest

from app.services.generation.mock_provider import MockLLMProvider
from app.services.generation.rag_service import RAGService


@pytest.fixture
def retrieval_results():
    return [
        {
            "filename": "PUBLIC MONEY.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "The project has an estimated cost of Rs. 10 lakh.",
        },
        {
            "filename": "PUBLIC MONEY.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "text": "The project was completed in March 2026.",
        },
    ]


def test_rag_service_generates_answer(retrieval_results):
    provider = MockLLMProvider(
        response="The project cost is Rs. 10 lakh."
    )

    service = RAGService(provider)

    answer = service.generate_answer(
        question="What is the project cost?",
        retrieval_results=retrieval_results,
    )

    assert answer == "The project cost is Rs. 10 lakh."


def test_rag_service_passes_grounded_prompt_to_provider(
    retrieval_results,
):
    provider = MockLLMProvider()

    service = RAGService(provider)

    service.generate_answer(
        question="What is the project cost?",
        retrieval_results=retrieval_results,
    )

    assert provider.response == "Mock response"


def test_rag_service_rejects_empty_question(retrieval_results):
    provider = MockLLMProvider()
    service = RAGService(provider)

    with pytest.raises(
        ValueError,
        match="Question must not be empty",
    ):
        service.generate_answer(
            question="   ",
            retrieval_results=retrieval_results,
        )


def test_rag_service_rejects_empty_retrieval_results():
    provider = MockLLMProvider()
    service = RAGService(provider)

    with pytest.raises(
        ValueError,
        match="Retrieval results must not be empty",
    ):
        service.generate_answer(
            question="What is the project cost?",
            retrieval_results=[],
        )