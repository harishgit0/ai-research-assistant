import pytest

from app.services.generation.gemini_provider import GeminiProvider


def test_gemini_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="GEMINI_API_KEY is not configured",
    ):
        GeminiProvider()


def test_gemini_provider_uses_environment_configuration(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    provider = GeminiProvider()

    assert provider.model == "test-model"


def test_gemini_provider_rejects_empty_prompt(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    provider = GeminiProvider()

    with pytest.raises(
        ValueError,
        match="Prompt must not be empty",
    ):
        provider.generate("   ")