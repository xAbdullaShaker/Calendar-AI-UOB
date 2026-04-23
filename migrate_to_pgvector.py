"""
migrate_to_pgvector.py — One-time migration: JSON embeddings → Supabase pgvector.

This script takes the locally generated embedding JSON files and uploads them
to Supabase PostgreSQL so the live server can search them using pgvector.

Run this once after generating embeddings, or any time the embeddings change.

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

# Load environment variables from .env file
load_dotenv()

# How many rows to insert per API call — avoids hitting Supabase request size limits
BATCH_SIZE = 50


def vec_str(embedding: list) -> str:
    """
    Convert a Python list of floats into the string format pgvector expects.
    Example: [0.1, 0.2, 0.3] → '[0.10000000,0.20000000,0.30000000]'
    pgvector requires this exact format when inserting vectors as text.
    """
    return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"


def migrate():
    """
    Main migration function — uploads FAQ and calendar embeddings to Supabase.

    Steps:
    1. Connect to Supabase using credentials from .env
    2. Clear the existing faq_embeddings table and reload it
    3. Clear the existing calendar_chunks table and reload it
    """
    # Read Supabase connection details from environment
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

    # Create connection to Supabase
    client = create_client(url, key)
    print(f"Connected to Supabase: {url}")

    # ── FAQ embeddings ─────────────────────────────────────────────────────────

    # Load the locally generated FAQ embedding file
    print("Loading faq_embeddings.json...")
    with open("faq_embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    faqs = data["faqs"] if "faqs" in data else data

    # Wipe the table so we start fresh (safe re-run behaviour)
    print("Clearing faq_embeddings table...")
    client.table("faq_embeddings").delete().neq("id", 0).execute()

    # Build one row per (faq_id, embedding) pair.
    # Each FAQ entry has multiple question embeddings — one per question variant.
    rows = []
    for entry in faqs:
        faq_id = entry["id"]
        for emb in entry["embeddings"]:
            # Convert the float list to pgvector string format
            rows.append({"faq_id": faq_id, "embedding": vec_str(emb)})

    # Insert rows in batches to avoid request size limits
    print(f"Inserting {len(rows)} FAQ vectors in batches of {BATCH_SIZE}...")
    for i in range(0, len(rows), BATCH_SIZE):
        client.table("faq_embeddings").insert(rows[i:i + BATCH_SIZE]).execute()
        print(f"  {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")
    print(f"  Done — {len(faqs)} FAQ entries uploaded")

    # ── Calendar chunks ────────────────────────────────────────────────────────

    # Load the locally generated calendar embedding file
    print("Loading calendar_embeddings.json...")
    with open("calendar_embeddings.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Wipe the table so we start fresh
    print("Clearing calendar_chunks table...")
    client.table("calendar_chunks").delete().neq("id", 0).execute()

    # Build one row per calendar event — each row has the text and its vector
    rows = [{"chunk": c["chunk"], "embedding": vec_str(c["embedding"])} for c in chunks]

    # Insert calendar rows in batches
    print(f"Inserting {len(rows)} calendar chunk vectors in batches of {BATCH_SIZE}...")
    for i in range(0, len(rows), BATCH_SIZE):
        client.table("calendar_chunks").insert(rows[i:i + BATCH_SIZE]).execute()
        print(f"  {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")
    print(f"  Done — {len(chunks)} calendar chunks uploaded")

    print("\nMigration complete. pgvector is ready.")


if __name__ == "__main__":
    migrate()
