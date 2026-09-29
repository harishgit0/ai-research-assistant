# AI Research Assistant

An AI-powered research and study assistant built as a portfolio and learning project. The system is designed to explore how modern retrieval-augmented generation (RAG) systems can combine semantic search, BM25, hybrid retrieval, reranking, evidence selection, and evaluation to produce grounded answers from research documents.

> **Project status:** Early implementation — PDF ingestion, cleaning, token-based chunking, and local embedding generation are currently implemented and tested.

## What this project is about

The goal is to build an end-to-end research assistant that can answer questions from a user's documents while keeping the retrieval pipeline measurable and explainable.

The planned pipeline is:

~~~text
Documents
   ↓
PDF / Text Extraction
   ↓
Cleaning
   ↓
Chunking + Metadata
   ↓
Embeddings
   ↓
Vector Database (pgvector)
   ↓
                    ┌─ Semantic / Vector Search
User Query → Query ┤
                    └─ BM25
                         ↓
                   Hybrid Retrieval
                         ↓
                     Reranking
                         ↓
                    Evidence Selection
                         ↓
                       Context
                         ↓
                         LLM
                         ↓
              Answer + Supporting Citations
~~~

The project will also include a retrieval and generation evaluation layer so that retrieval quality and answer quality can be measured rather than assumed.

## Current implementation

### PDF ingestion

- Extracts text page-by-page using pypdf
- Preserves page numbers as metadata
- Handles pages with missing text safely

### Text cleaning

- Normalizes extracted text conservatively
- Removes isolated formatting artifacts
- Removes trailing page-number artifacts where appropriate
- Avoids aggressive cleaning that could remove useful research content

### Token-based chunking

- Uses tiktoken
- Default chunk size: **400 tokens**
- Default overlap: **75 tokens**
- Returns chunk index, text, and token count
- Validates that chunk overlap is smaller than chunk size

### Embeddings

Current embedding model:

**BAAI/bge-small-en-v1.5**

- 384-dimensional embeddings
- Runs locally
- CPU-compatible
- Retrieval-oriented embedding model
- Embeddings are normalized for similarity-based retrieval

### Testing

The current pipeline has tests covering:

- Chunking behavior
- Embedding dimensionality
- Multiple-text embedding shape
- Semantic similarity behavior
- End-to-end PDF → cleaning → chunking → embedding flow

The current sample document produces **6 chunks**, with embeddings of shape **(6, 384)** and normalized vector length approximately **1.0**.

## Planned retrieval system

The retrieval layer will compare and combine multiple retrieval strategies:

### Semantic retrieval

Use vector similarity between the query embedding and document-chunk embeddings stored in PostgreSQL with pgvector.

### BM25

Use lexical retrieval to capture exact terms, names, technical terminology, and keyword-heavy queries that semantic retrieval can miss.

### Hybrid retrieval

Combine semantic and lexical retrieval using a measurable fusion strategy rather than relying on one retrieval method alone.

### Reranking

A dedicated reranker will reorder the candidate chunks returned by the first-stage retrieval system.

### Evidence selection

The system will retain the most relevant supporting passages and their metadata so that generated answers can be traced back to source chunks and pages.

## Evaluation

A major goal of this project is to evaluate the retrieval pipeline quantitatively.

Planned retrieval metrics:

- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- NDCG

Planned generation/evidence metrics:

- Faithfulness
- Answer relevance
- Context relevance
- Citation correctness

The project will use evaluation results to compare retrieval approaches and document design decisions instead of claiming improvements without measurement.

## Tech stack

### Backend

- Python 3.12
- FastAPI
- Uvicorn

### Document processing

- pypdf
- tiktoken

### Retrieval / NLP

- Sentence Transformers
- BAAI/bge-small-en-v1.5
- BM25
- pgvector
- PostgreSQL

### Frontend

- Vue 3
- CSS

### Testing

- pytest

### Planned components

- Reranker
- LLM provider abstraction
- Hybrid retrieval
- Retrieval evaluation
- Answer/evidence evaluation

## Project structure

~~~text
ai-research-assistant/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── services/
│   │       ├── embedding.py
│   │       └── ingestion/
│   │           ├── cleaner.py
│   │           ├── chunker.py
│   │           └── pdf.py
│   │
│   ├── tests/
│   │   ├── test_chunker.py
│   │   ├── test_embeddings.py
│   │   ├── test_embedding_service.py
│   │   └── test_document_embeddings.py
│   │
│   └── data/
│       └── uploads/
│
├── .gitignore
└── README.md
~~~

Uploaded documents are intentionally excluded from Git through .gitignore.

## Running the current backend

Create and activate the backend virtual environment:

~~~bash
cd backend
source .venv/bin/activate
~~~

Install the project dependencies as they are added to the project.

Run the current test suite:

~~~bash
pytest
~~~

The current implementation is primarily exercised through the test suite while the application architecture is being built incrementally.

## Roadmap

- [x] PDF text extraction
- [x] Text cleaning
- [x] Token-based chunking
- [x] BGE embedding service
- [x] Embedding and pipeline tests
- [ ] PostgreSQL + pgvector storage
- [ ] Vector similarity retrieval
- [ ] BM25 retrieval
- [ ] Hybrid retrieval
- [ ] Reranking
- [ ] Evidence selection and source metadata
- [ ] LLM provider abstraction
- [ ] RAG answer generation
- [ ] Citation-aware answers
- [ ] Retrieval evaluation dataset
- [ ] Retrieval metrics and comparison experiments
- [ ] Generation/evidence evaluation
- [ ] Vue 3 research assistant interface
- [ ] Final documentation and architecture walkthrough

## Why this project?

This project is intentionally built around the internals of a modern RAG system rather than only wrapping an LLM API.

The focus is on understanding and demonstrating:

- How documents become searchable representations
- Why chunking affects retrieval
- How semantic and lexical retrieval differ
- Why hybrid retrieval can be useful
- What reranking contributes after first-stage retrieval
- How evidence can be traced to source documents
- How retrieval quality can be evaluated quantitatively
- How grounded generation can be tested rather than assumed

The end goal is a technically explainable AI research assistant that can also serve as a practical demonstration of information retrieval, NLP, embeddings, RAG, and evaluation concepts.
