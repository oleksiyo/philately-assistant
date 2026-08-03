"""Postgres logging of conversations and user feedback for the RAG assistant."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime

import psycopg2
from dotenv import load_dotenv

load_dotenv()

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    retrieval_method TEXT NOT NULL,
    prompt_variant TEXT NOT NULL,
    model TEXT NOT NULL,
    latency_seconds DOUBLE PRECISION NOT NULL,
    sources JSONB NOT NULL,
    feedback SMALLINT
);
"""


def _connection_params() -> dict:
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
        "dbname": os.environ.get("POSTGRES_DB", "philately"),
        "user": os.environ.get("POSTGRES_USER", "philately"),
        "password": os.environ.get("POSTGRES_PASSWORD", "philately"),
    }


@contextmanager
def get_connection():
    conn = psycopg2.connect(**_connection_params())
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        conn.commit()


def log_conversation(
    question: str,
    answer: str,
    retrieval_method: str,
    prompt_variant: str,
    model: str,
    latency_seconds: float,
    sources: list[dict],
    created_at: datetime | None = None,
) -> int:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations
                (question, answer, retrieval_method, prompt_variant, model,
                 latency_seconds, sources, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, COALESCE(%s, now()))
            RETURNING id
            """,
            (
                question,
                answer,
                retrieval_method,
                prompt_variant,
                model,
                latency_seconds,
                json.dumps(sources),
                created_at,
            ),
        )
        conversation_id = cur.fetchone()[0]
        conn.commit()
        return conversation_id


def log_feedback(conversation_id: int, feedback: int) -> None:
    if feedback not in (-1, 1):
        raise ValueError(f"feedback must be -1 or 1, got {feedback!r}")
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE conversations SET feedback = %s WHERE id = %s",
            (feedback, conversation_id),
        )
        conn.commit()
