import os

from dotenv import load_dotenv
from google import genai

from app.services.generation.llm_provider import LLMProvider


load_dotenv()


class GeminiProvider(LLMProvider):
    """Gemini implementation of the LLM provider interface."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash",
        )

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text.strip()