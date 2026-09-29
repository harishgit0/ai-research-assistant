import tiktoken


DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 75


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping token-based chunks.

    Returns a list of dictionaries containing:
    - chunk_index
    - text
    - token_count
    """

    if not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    tokenizer = tiktoken.get_encoding("cl100k_base")

    tokens = tokenizer.encode(text)

    chunks = []

    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))

        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text.strip(),
            "token_count": len(chunk_tokens),
        })

        chunk_index += 1

        if end == len(tokens):
            break

        start = end - chunk_overlap

    return chunks
