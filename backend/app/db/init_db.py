"""Initialize the PostgreSQL schema. Run from backend/ with PYTHONPATH=."""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")


def initialize_database() -> None:
    required = ("AI_RESEARCH_DB_NAME", "AI_RESEARCH_DB_USER",
                "AI_RESEARCH_DB_PASSWORD", "AI_RESEARCH_DB_HOST",
                "AI_RESEARCH_DB_PORT")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing database configuration: " + ", ".join(missing)
            + ". Copy .env.example to .env and configure it."
        )

    conn = psycopg.connect(
        dbname=os.environ["AI_RESEARCH_DB_NAME"],
        user=os.environ["AI_RESEARCH_DB_USER"],
        password=os.environ["AI_RESEARCH_DB_PASSWORD"],
        host=os.environ["AI_RESEARCH_DB_HOST"],
        port=os.environ["AI_RESEARCH_DB_PORT"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    embedding vector(384),
                    UNIQUE (document_id, chunk_index)
                );
            """)
            cur.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_at "
                "TIMESTAMPTZ NOT NULL DEFAULT NOW();"
            )
            cur.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS token_count "
                "INTEGER NOT NULL DEFAULT 0;"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_document_id_idx "
                "ON chunks(document_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx "
                "ON chunks USING hnsw (embedding vector_cosine_ops);"
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    print("Database schema initialized successfully.")


if __name__ == "__main__":
    initialize_database()
