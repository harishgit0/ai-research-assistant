from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException

from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation.gemini_provider import GeminiProvider
from app.services.generation.rag_service import RAGService
from app.services.ingestion.pdf import extract_text_from_pdf
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.chunker import chunk_text
from app.services.research_assistant_service import ResearchAssistantService
app = FastAPI(title="AI Research Assistant")

gemini_provider = GeminiProvider()
rag_service = RAGService(
    llm_provider=gemini_provider,
)
research_assistant = ResearchAssistantService(
    rag_service=rag_service,
)


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "AI Research Assistant API is running"
    }


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_path = UPLOAD_DIR / file.filename

    contents = await file.read()

    file_path.write_bytes(contents)

    pages = extract_text_from_pdf(str(file_path))

    chunks = []

    for page in pages:
        page["text"] = clean_text(
            page["text"],
            page_number=page["page_number"]
        )

        page_chunks = chunk_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "page_number": page["page_number"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "token_count": chunk["token_count"],
            })
    return {
        "filename": file.filename,
        "pages": len(pages),
        "chunks": chunks,
    }


@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    try:
        result = research_assistant.answer(
            question=request.question,
        )

        sources = [
            {
                "filename": source["filename"],
                "page_number": source["page_number"],
                "chunk_index": source["chunk_index"],
                "reranker_score": source["reranker_score"],
                "text": source["text"],
            }
            for source in result["sources"]
        ]

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": sources,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc