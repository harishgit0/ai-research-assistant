class PromptBuilder:
    """Build a deterministic RAG prompt from a question and retrieved context."""

    def build(self, question: str, context: str) -> str:
        """Create a grounded prompt for an LLM using retrieved evidence."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        if not context or not context.strip():
            raise ValueError("Context must not be empty.")

        return "\n".join(
            [
                "You are a research assistant.",
                "Answer the user's question using only the provided sources.",
                "If the sources do not contain enough information, say that the answer cannot be determined from the provided sources.",
                "Do not invent facts or citations.",
                "",
                "SOURCES:",
                context,
                "",
                "QUESTION:",
                question.strip(),
                "",
                "ANSWER:",
            ]
        )
