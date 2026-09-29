import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def get_connection():
    return psycopg.connect(
        dbname=os.environ["AI_RESEARCH_DB_NAME"],
        user=os.environ["AI_RESEARCH_DB_USER"],
        password=os.environ["AI_RESEARCH_DB_PASSWORD"],
        host=os.environ["AI_RESEARCH_DB_HOST"],
        port=os.environ["AI_RESEARCH_DB_PORT"],
    )
