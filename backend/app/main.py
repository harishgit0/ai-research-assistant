import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.db.connection import get_connection
from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation.gemini_provider import GeminiProvider
from app.services.generation.mock_provider import MockLLMProvider
from app.services.generation.rag_service import RAGService
from app.services.ingestion.indexer import index_pdf
from app.services.research_assistant_service import ResearchAssistantService


BACKEND_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BACKEND_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

app = FastAPI(
    title="AI Research Assistant",
    description="Document-grounded research assistant with hybrid retrieval and citations.",
    version="1.0.0",
)
origins = [
    value.strip()
    for value in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@lru_cache(maxsize=1)
def get_research_assistant() -> ResearchAssistantService:
    """Construct generation services on demand, so the API starts without secrets."""
    provider_name = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider_name == "mock":
        provider = MockLLMProvider(
            "Mock mode is enabled. Configure GEMINI_API_KEY and set "
            "LLM_PROVIDER=gemini to generate answers from retrieved evidence."
        )
    elif provider_name == "gemini":
        provider = GeminiProvider()
    else:
        raise ValueError("LLM_PROVIDER must be either 'gemini' or 'mock'.")
    return ResearchAssistantService(rag_service=RAGService(llm_provider=provider))


@app.get("/")
def root():
    return {"name": "AI Research Assistant", "message": "API is running.",
            "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        finally:
            conn.close()
        return {"status": "ok", "database": "connected"}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Check .env and initialize the schema.",
        ) from exc


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")
    filename = Path(file.filename).name
    if filename in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the {MAX_UPLOAD_BYTES}-byte upload limit.",
        )
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / filename
    path.write_bytes(contents)
    try:
        count = index_pdf(str(path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Indexing failed. Check database configuration and server logs.",
        ) from exc

    return {"filename": filename, "message": "Document indexed successfully.",
            "chunks": count}


@app.get("/documents")
def list_documents():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT d.id, d.filename, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id, d.filename
                ORDER BY d.id DESC;
            """)
            rows = cur.fetchall()
        return [{"id": row[0], "filename": row[1], "chunk_count": row[2]}
                for row in rows]
    finally:
        conn.close()


@app.delete("/documents/{document_id}")
def delete_document(document_id: int):
    conn = get_connection()
    stored_path = None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT file_path FROM documents WHERE id = %s;", (document_id,))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Document not found.")
            stored_path = row[0]
            cur.execute("DELETE FROM documents WHERE id = %s;", (document_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # Remove only files located within the application's upload directory.
    if stored_path:
        try:
            resolved = Path(stored_path).resolve()
            resolved.relative_to(UPLOAD_DIR.resolve())
            resolved.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
    return {"message": "Document and its indexed chunks were deleted."}


@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    if request.document_id is not None:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM documents WHERE id = %s;", (request.document_id,))
                exists = cur.fetchone() is not None
        finally:
            conn.close()
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"Document with id {request.document_id} was not found.",
            )

    try:
        result = get_research_assistant().answer(
            question=request.question,
            document_id=request.document_id,
        )
    except ValueError as exc:
        if "GEMINI_API_KEY" in str(exc):
            raise HTTPException(
                status_code=503,
                detail="Gemini is not configured. Set GEMINI_API_KEY or use LLM_PROVIDER=mock.",
            ) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail="The answer provider failed. Check its configuration and server logs.",
        ) from exc

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": [
            {
                "filename": source["filename"],
                "page_number": source["page_number"],
                "chunk_index": source["chunk_index"],
                "reranker_score": source["reranker_score"],
                "text": source["text"],
            }
            for source in result["sources"]
        ],
    }
