import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def get_connection():
    """Open PostgreSQL using the repository's environment configuration."""
    required = ("AI_RESEARCH_DB_NAME", "AI_RESEARCH_DB_USER",
                "AI_RESEARCH_DB_PASSWORD", "AI_RESEARCH_DB_HOST",
                "AI_RESEARCH_DB_PORT")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing database configuration: " + ", ".join(missing)
            + ". Copy .env.example to .env and configure it."
        )
    return psycopg.connect(
        dbname=os.environ["AI_RESEARCH_DB_NAME"],
        user=os.environ["AI_RESEARCH_DB_USER"],
        password=os.environ["AI_RESEARCH_DB_PASSWORD"],
        host=os.environ["AI_RESEARCH_DB_HOST"],
        port=os.environ["AI_RESEARCH_DB_PORT"],
    )
