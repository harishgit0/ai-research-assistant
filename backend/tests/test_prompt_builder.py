import pytest

from app.services.generation.prompt_builder import PromptBuilder


def test_build_prompt_contains_instructions_context_and_question():
    builder = PromptBuilder()

    prompt = builder.build(
        "What is retrieval-augmented generation?",
        "[Source 1]\nDocument: paper.pdf\nPage: 2\nChunk: 0\n\nRAG combines retrieval with generation.",
    )

    assert "You are a research assistant." in prompt
    assert "Answer the user's question using only the provided sources." in prompt
    assert "SOURCES:" in prompt
    assert "RAG combines retrieval with generation." in prompt
    assert "QUESTION:\nWhat is retrieval-augmented generation?" in prompt
    assert prompt.endswith("ANSWER:")


def test_build_prompt_preserves_context_content_and_order():
    builder = PromptBuilder()

    context = (
        "[Source 1]\nDocument: first.pdf\nPage: 1\nChunk: 0\n\nFirst source."
        "\n\n"
        "[Source 2]\nDocument: second.pdf\nPage: 4\nChunk: 2\n\nSecond source."
    )

    prompt = builder.build("What do the sources say?", context)

    assert prompt.index("First source.") < prompt.index("Second source.")
    assert prompt.index("SOURCES:") < prompt.index("First source.")
    assert prompt.index("Second source.") < prompt.index("QUESTION:")


def test_build_prompt_strips_question_whitespace():
    builder = PromptBuilder()

    prompt = builder.build("  What is RAG?  ", "Evidence")

    assert "QUESTION:\nWhat is RAG?" in prompt


def test_build_prompt_rejects_empty_question():
    builder = PromptBuilder()

    with pytest.raises(ValueError, match="Question must not be empty"):
        builder.build("   ", "Evidence")


def test_build_prompt_rejects_empty_context():
    builder = PromptBuilder()

    with pytest.raises(ValueError, match="Context must not be empty"):
        builder.build("What is RAG?", "   ")
