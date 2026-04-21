"""
migrate_to_pgvector.py — One-time migration: JSON embeddings → Supabase pgvector.

Prerequisites:
  1. Run the SQL in the Supabase SQL Editor to create tables and RPC functions.
  2. Set SUPABASE_URL and SUPABASE_KEY in .env.
  3. Run: python migrate_to_pgvector.py

Safe to re-run — clears and reloads both tables each time.
"""

import json
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 50


def vec_str(embedding: list) -> str:
    """Format embedding list as pgvector string '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"


def migrate():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

    client = create_client(url, key)
    print(f"Connected to Supabase: {url}")

    # ── FAQ embeddings ─────────────────────────────────────────────────────────
    print("Loading faq_embeddings.json...")
    with open("faq_embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    faqs = data["faqs"] if "faqs" in data else data

    print("Clearing faq_embeddings table...")
    client.table("faq_embeddings").delete().neq("id", 0).execute()

    rows = []
    for entry in faqs:
        faq_id = entry["id"]
        for emb in entry["embeddings"]:
            rows.append({"faq_id": faq_id, "embedding": vec_str(emb)})

    print(f"Inserting {len(rows)} FAQ vectors in batches of {BATCH_SIZE}...")
    for i in range(0, len(rows), BATCH_SIZE):
        client.table("faq_embeddings").insert(rows[i:i + BATCH_SIZE]).execute()
        print(f"  {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")
    print(f"  Done — {len(faqs)} FAQ entries uploaded")

    # ── Calendar chunks ────────────────────────────────────────────────────────
    print("Loading calendar_embeddings.json...")
    with open("calendar_embeddings.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Clearing calendar_chunks table...")
    client.table("calendar_chunks").delete().neq("id", 0).execute()

    rows = [{"chunk": c["chunk"], "embedding": vec_str(c["embedding"])} for c in chunks]
    print(f"Inserting {len(rows)} calendar chunk vectors in batches of {BATCH_SIZE}...")
    for i in range(0, len(rows), BATCH_SIZE):
        client.table("calendar_chunks").insert(rows[i:i + BATCH_SIZE]).execute()
        print(f"  {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")
    print(f"  Done — {len(chunks)} calendar chunks uploaded")

    print("\nMigration complete. pgvector is ready.")


if __name__ == "__main__":
    migrate()
