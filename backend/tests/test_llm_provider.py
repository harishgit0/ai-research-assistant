import pytest

from app.services.generation.llm_provider import LLMProvider
from app.services.generation.mock_provider import MockLLMProvider


class TestProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        return f"generated: {prompt}"


def test_provider_contract_can_be_implemented():
    provider = TestProvider()

    assert provider.generate("hello") == "generated: hello"


def test_mock_provider_returns_deterministic_response():
    provider = MockLLMProvider("test response")

    assert provider.generate("What is RAG?") == "test response"


def test_mock_provider_rejects_empty_prompt():
    provider = MockLLMProvider()

    with pytest.raises(ValueError, match="Prompt must not be empty"):
        provider.generate("   ")


def test_llm_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()
