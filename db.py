"""
db.py — PostgreSQL pgvector connection pool and vector search queries.

Requires DATABASE_URL in .env pointing to a Postgres instance with pgvector.
Free options: Supabase (supabase.com) or Neon (neon.tech) — both have pgvector.
"""

import os
import numpy as np
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from pgvector.psycopg2 import register_vector

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL is not set.")
        _pool = ThreadedConnectionPool(1, 10, url)
    return _pool


def _get_conn():
    conn = _get_pool().getconn()
    register_vector(conn)
    return conn


def _release_conn(conn):
    _get_pool().putconn(conn)


# ── Queries ───────────────────────────────────────────────────────────────────

def find_best_faq_match_db(embedding: list) -> tuple[str, float]:
    """Return (faq_id, cosine_similarity_score) for the closest FAQ vector."""
    vec = np.array(embedding, dtype=np.float32)
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT faq_id,
                       1 - (embedding <=> %s::vector) AS score
                FROM   faq_embeddings
                ORDER  BY embedding <=> %s::vector
                LIMIT  1
                """,
                (vec, vec),
            )
            row = cur.fetchone()
            return (row[0], float(row[1])) if row else (None, 0.0)
    finally:
        _release_conn(conn)


def retrieve_top_chunks_db(embedding: list, top_k: int = 4) -> list[str]:
    """Return the top_k most similar calendar chunk texts."""
    vec = np.array(embedding, dtype=np.float32)
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk
                FROM   calendar_chunks
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
                """,
                (vec, top_k),
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        _release_conn(conn)
