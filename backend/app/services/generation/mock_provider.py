from app.services.generation.llm_provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Deterministic provider used for testing the generation pipeline."""

    def __init__(self, response: str = "Mock response"):
        self.response = response

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        return self.response
