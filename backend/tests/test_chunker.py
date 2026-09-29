from app.services.ingestion.chunker import chunk_text


def test_chunk_text():
    text = "Machine learning is " * 300

    chunks = chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk["token_count"] <= 100
        assert chunk["text"]


def test_empty_text():
    chunks = chunk_text("")

    assert chunks == []


def test_invalid_overlap():
    try:
        chunk_text(
            "Some text",
            chunk_size=100,
            chunk_overlap=100,
        )
        assert False
    except ValueError:
        assert True
