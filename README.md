# AI Research Assistant

An end-to-end research assistant for asking questions about PDF papers and receiving answers grounded in retrieved passages. It demonstrates document ingestion, local embeddings, semantic and lexical retrieval, hybrid ranking, cross-encoder reranking, source-aware RAG prompts, and retrieval evaluation.

## Features

- Upload and index PDF papers; extract page text, clean it, and chunk it (400 tokens with 75-token overlap).
- Generate normalized 384-dimensional embeddings with BAAI/bge-small-en-v1.5.
- Store documents, chunks, and embeddings in PostgreSQL with pgvector.
- Combine semantic search and BM25 with Reciprocal Rank Fusion (RRF).
- Rerank candidates with BAAI/bge-reranker-base.
- Generate grounded answers with Gemini and return source filename, page, chunk index, and text.
- Search one selected document or the full library; delete indexed documents.
- Vue 3 frontend for upload, question answering, library management, and evidence inspection.
- Retrieval evaluation with Recall@K, MRR, and NDCG@K.

## Architecture

    PDF -> extraction -> cleaning -> token chunks -> BGE embeddings -> PostgreSQL/pgvector
    Question -> semantic search + BM25 -> RRF -> cross-encoder reranking
             -> context builder -> grounded prompt -> Gemini -> answer + source passages

## Stack

Python 3.12+, FastAPI, Uvicorn, PostgreSQL, pgvector, psycopg, Sentence Transformers, BM25, Google Gen AI SDK, Vue 3, Vite, pytest.

## Prerequisites

- Python 3.12+
- Node.js 20+ and npm
- PostgreSQL with pgvector available
- Gemini API key for generated answers (optional for startup; see mock mode)

Embedding and reranker models download from Hugging Face on first use and may require several hundred megabytes of storage and memory.

## Setup

### 1. Configure environment

From the repository root:

    cp .env.example .env

Set your database credentials in .env and add GEMINI_API_KEY for real generated answers. Never commit this file. Create the database if needed, for example:

    createdb ai_research_assistant

Your PostgreSQL role needs permission to enable the pgvector extension, or an administrator must enable it first.

### 2. Install backend dependencies

    cd backend
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

PowerShell activation: .venv\Scripts\Activate.ps1

### 3. Initialize database

From backend/ with the virtual environment active:

    PYTHONPATH=. python -m app.db.init_db

This creates the extension if permitted, the documents and chunks tables, and supporting indexes. It is designed to be repeatable.

### 4. Start API

From backend/:

    PYTHONPATH=. uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000/docs for interactive API docs.

- GET / and GET /health — service and database status
- GET /documents — indexed documents
- POST /documents/upload — upload a PDF using multipart field file
- DELETE /documents/{document_id} — remove a document and its indexed chunks
- POST /query — ask a question

Example JSON body for POST /query:

    {
      "question": "What are the main findings?",
      "document_id": 1
    }

Use null for document_id to search across all indexed documents.

### 5. Start Vue frontend

Open a second terminal from the repository root:

    cd frontend
    npm install
    npm run dev

Open http://127.0.0.1:5173. The frontend defaults to API http://127.0.0.1:8000. To change it, create frontend/.env.local containing:

    VITE_API_BASE_URL=https://your-api-host.example

Set CORS_ORIGINS in the API environment to include the frontend origin when running outside localhost.

## LLM configuration

Gemini is the default provider. Configure GEMINI_API_KEY and optionally GEMINI_MODEL in .env. To test the UI request flow without a key, set:

    LLM_PROVIDER=mock

Mock mode returns a fixed explanatory response, not an LLM-generated answer. It is for smoke-testing only. Restart the API after changing environment variables.

## Tests

From backend/ with the virtual environment active:

    PYTHONPATH=. pytest

Some integration tests require PostgreSQL and locally downloaded embedding/reranker models. Optional sample-PDF tests skip when the files are absent.

## Retrieval evaluation

From backend/:

    PYTHONPATH=. python -m app.services.evaluation.run_evaluation
    PYTHONPATH=. python -m app.services.evaluation.run_evaluation --list-chunks RAG-Survey.pdf
    PYTHONPATH=. python -m app.services.evaluation.run_evaluation --rrf-sweep

The evaluation dataset has manually assigned chunk indexes tied to a particular PDF extraction and chunking result. Validate each relevant chunk against the exact indexed PDF before treating metrics as a benchmark. Do not infer gold chunk IDs from question wording. Retrieval metrics measure ranking against supplied labels; they do not independently measure answer faithfulness or factual correctness.

## Limitations

- Uploads are limited to 25 MiB by default; change MAX_UPLOAD_BYTES to adjust the limit.
- Scanned/image-only PDFs need OCR, which is not included.
- Models run locally; first use can be slow and memory-intensive.
- Gemini use is subject to API availability and quotas.
- This is a portfolio MVP, not a hardened multi-user service. It has no user accounts, per-user document isolation, background indexing queue, OCR, or deployment configuration.
- Uploaded PDFs are stored locally under backend/data/uploads/; do not expose this directory publicly.

## Project layout

    backend/app/db/           PostgreSQL connection and schema initialization
    backend/app/services/     ingestion, retrieval, generation, evaluation
    backend/app/main.py       FastAPI endpoints
    backend/tests/            unit and integration tests
    frontend/src/App.vue      document library, upload, query, source UI
    frontend/src/style.css    responsive dark interface
    .env.example              environment template
    backend/requirements.txt  backend dependencies

The goal is to demonstrate the internals of a practical RAG system rather than only forwarding prompts to an LLM API: preprocessing, chunking, embedding models, retrieval, fusion, reranking, source attribution, and evaluation.
