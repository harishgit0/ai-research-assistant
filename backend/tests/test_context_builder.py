from app.services.generation.context_builder import ContextBuilder


def test_build_context_formats_sources_and_metadata():
    results = [
        {
            "filename": "RAG-Survey.pdf",
            "page_number": 1,
            "chunk_index": 11,
            "text": "RAG combines retrieval with generation.",
        },
        {
            "filename": "RAG-Survey.pdf",
            "page_number": 17,
            "chunk_index": 69,
            "text": "A retriever supplies relevant evidence.",
        },
    ]

    context = ContextBuilder().build(results)

    assert context == (
        "[Source 1]\n"
        "Document: RAG-Survey.pdf\n"
        "Page: 1\n"
        "Chunk: 11\n"
        "\n"
        "RAG combines retrieval with generation.\n\n"
        "[Source 2]\n"
        "Document: RAG-Survey.pdf\n"
        "Page: 17\n"
        "Chunk: 69\n"
        "\n"
        "A retriever supplies relevant evidence."
    )


def test_build_context_preserves_result_order():
    results = [
        {
            "filename": "first.pdf",
            "page_number": 2,
            "chunk_index": 3,
            "text": "First evidence.",
        },
        {
            "filename": "second.pdf",
            "page_number": 4,
            "chunk_index": 8,
            "text": "Second evidence.",
        },
    ]

    context = ContextBuilder().build(results)

    assert context.index("First evidence.") < context.index("Second evidence.")
    assert context.index("[Source 1]") < context.index("[Source 2]")


def test_build_context_preserves_text_verbatim():
    text = "Exact text:  A/B + C?\nDo not rewrite this."
    results = [
        {
            "filename": "notes.pdf",
            "page_number": 3,
            "chunk_index": 0,
            "text": text,
        }
    ]

    context = ContextBuilder().build(results)

    assert text in context


def test_build_context_returns_empty_string_for_no_results():
    assert ContextBuilder().build([]) == ""
