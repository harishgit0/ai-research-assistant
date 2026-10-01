from app.services.generation.context_builder import ContextBuilder
from app.services.generation.llm_provider import LLMProvider
from app.services.generation.prompt_builder import PromptBuilder


class RAGService:
    """Orchestrate context building, prompt construction, and LLM generation."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
    ):
        self.llm_provider = llm_provider
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate_answer(
        self,
        question: str,
        retrieval_results: list[dict],
    ) -> str:
        """Generate an answer from retrieved evidence."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        if not retrieval_results:
            raise ValueError("Retrieval results must not be empty.")

        context = self.context_builder.build(retrieval_results)

        prompt = self.prompt_builder.build(
            question=question,
            context=context,
        )

        return self.llm_provider.generate(prompt)