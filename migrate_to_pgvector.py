"""
migrate_to_pgvector.py — One-time migration: JSON embeddings → PostgreSQL pgvector.

Run once after setting DATABASE_URL in .env:
    python migrate_to_pgvector.py

Safe to re-run — clears and reloads both tables each time.
"""

import json
import os
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()


def migrate():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set in .env")

    conn = psycopg2.connect(url)
    register_vector(conn)
    cur = conn.cursor()

    print("Setting up pgvector extension and tables...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS faq_embeddings (
            id        SERIAL PRIMARY KEY,
            faq_id    TEXT NOT NULL,
            embedding vector(1024)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS calendar_chunks (
            id        SERIAL PRIMARY KEY,
            chunk     TEXT NOT NULL,
            embedding vector(1024)
        )
    """)

    # ── FAQ embeddings ─────────────────────────────────────────────────────────
    print("Loading faq_embeddings.json...")
    with open("faq_embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    faqs = data["faqs"] if "faqs" in data else data

    cur.execute("DELETE FROM faq_embeddings")
    count = 0
    for entry in faqs:
        faq_id = entry["id"]
        for emb in entry["embeddings"]:
            cur.execute(
                "INSERT INTO faq_embeddings (faq_id, embedding) VALUES (%s, %s)",
                (faq_id, np.array(emb, dtype=np.float32)),
            )
            count += 1
    print(f"  {count} FAQ vectors inserted ({len(faqs)} entries)")

    # ── Calendar chunks ────────────────────────────────────────────────────────
    print("Loading calendar_embeddings.json...")
    with open("calendar_embeddings.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    cur.execute("DELETE FROM calendar_chunks")
    for c in chunks:
        cur.execute(
            "INSERT INTO calendar_chunks (chunk, embedding) VALUES (%s, %s)",
            (c["chunk"], np.array(c["embedding"], dtype=np.float32)),
        )
    print(f"  {len(chunks)} calendar chunk vectors inserted")

    # ── Indexes ────────────────────────────────────────────────────────────────
    print("Building indexes...")
    cur.execute("DROP INDEX IF EXISTS faq_embeddings_cosine_idx")
    cur.execute("DROP INDEX IF EXISTS calendar_chunks_cosine_idx")
    cur.execute("""
        CREATE INDEX faq_embeddings_cosine_idx
        ON faq_embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10)
    """)
    cur.execute("""
        CREATE INDEX calendar_chunks_cosine_idx
        ON calendar_chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10)
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Migration complete. pgvector is ready.")


if __name__ == "__main__":
    migrate()
