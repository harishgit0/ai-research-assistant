from app.services.generation.gemini_provider import GeminiProvider
from app.services.generation.rag_service import RAGService
from app.services.research_assistant_service import ResearchAssistantService


def main():
    question = "What is retrieval augmented generation?"

    provider = GeminiProvider()

    rag_service = RAGService(
        llm_provider=provider,
    )

    assistant = ResearchAssistantService(
        rag_service=rag_service,
    )

    result = assistant.answer(
        question=question,
        retrieval_k=20,
        top_k=5,
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for index, source in enumerate(result["sources"], start=1):
        print(
            f"\n[{index}] "
            f"{source['filename']} "
            f"| Page {source['page_number']} "
            f"| Chunk {source['chunk_index']} "
            f"| Reranker Score {source['reranker_score']:.6f}"
        )

        print(source["text"][:500].replace("\n", " "))


if __name__ == "__main__":
    main()