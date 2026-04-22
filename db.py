"""
db.py — Supabase vector search via HTTPS (supabase-py).

Requires SUPABASE_URL and SUPABASE_KEY in .env.
Uses RPC functions match_faq and match_calendar (created via SQL Editor).
"""

import os
from supabase import create_client, Client

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        _client = create_client(url, key)
    return _client


def find_best_faq_match_db(embedding: list) -> tuple[str, float]:
    """Return (faq_id, cosine_similarity_score) for the closest FAQ vector."""
    try:
        result = _get_client().rpc("match_faq", {
            "query_embedding": embedding,
            "match_threshold": 0.0,
            "match_count": 1,
        }).execute()
        if result.data:
            row = result.data[0]
            return (row["faq_id"], float(row["similarity"]))
    except Exception as e:
        print(f"[DB find_best_faq_match_db ERROR] {e}")
    return (None, 0.0)


def retrieve_top_chunks_db(embedding: list, top_k: int = 4) -> list[str]:
    """Return the top_k most similar calendar chunk texts."""
    try:
        result = _get_client().rpc("match_calendar", {
            "query_embedding": embedding,
            "match_count": top_k,
        }).execute()
        return [row["chunk"] for row in result.data] if result.data else []
    except Exception as e:
        print(f"[DB retrieve_top_chunks_db ERROR] {e}")
        return []
