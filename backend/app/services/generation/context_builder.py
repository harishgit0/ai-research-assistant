class ContextBuilder:
    """Build a deterministic evidence context from retrieval results."""

    def build(self, results: list[dict]) -> str:
        """Format retrieved chunks as source-labeled evidence for a prompt."""
        if not results:
            return ""

        sources = []

        for index, result in enumerate(results, start=1):
            sources.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"Document: {result['filename']}",
                        f"Page: {result['page_number']}",
                        f"Chunk: {result['chunk_index']}",
                        "",
                        result["text"],
                    ]
                )
            )

        return "\n\n".join(sources)
