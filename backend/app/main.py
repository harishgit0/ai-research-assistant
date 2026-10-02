from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException

from app.db.connection import get_connection
from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation.gemini_provider import GeminiProvider
from app.services.generation.rag_service import RAGService
from app.services.ingestion.indexer import index_pdf
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

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required."
        )

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty."
        )

    file_path.write_bytes(contents)

    try:
        chunk_count = index_pdf(str(file_path))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "filename": safe_filename,
        "message": "Document indexed successfully.",
        "chunks": chunk_count,
    }


@app.get("/documents")
def list_documents():
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    d.id,
                    d.filename,
                    COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id, d.filename
                ORDER BY d.id DESC;
                """
            )

            rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "filename": row[1],
                "chunk_count": row[2],
            }
            for row in rows
        ]
    finally:
        connection.close()


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
