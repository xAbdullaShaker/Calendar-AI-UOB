"""
db.py — Supabase vector search via HTTPS (supabase-py).

This file handles all database communication with Supabase.
Instead of searching vectors in Python (slow, uses RAM), it sends the
query vector to Supabase and lets PostgreSQL do the search using pgvector.

Requires SUPABASE_URL and SUPABASE_KEY in .env.
Uses RPC functions match_faq and match_calendar (created via SQL Editor).
"""

import os
from supabase import create_client, Client

# Global variable to hold the Supabase client — created once and reused
_client: Client | None = None


def _get_client() -> Client:
    """
    Return the Supabase client, creating it on first call.
    This is called 'lazy initialization' — we only connect when we actually need to.
    Raises an error if the required environment variables are missing.
    """
    global _client
    if _client is None:
        # Read connection details from environment variables
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        # Create and store the client for future calls
        _client = create_client(url, key)
    return _client


def find_best_faq_match_db(embedding: list) -> tuple[str, float]:
    """
    Search the FAQ vectors in Supabase and return the closest match.

    Calls the 'match_faq' SQL function which uses pgvector's cosine distance
    operator (<=>). Returns the FAQ entry id and its similarity score (0.0–1.0).
    Returns (None, 0.0) if the search fails.
    """
    try:
        # Call the match_faq stored procedure in Supabase with the query vector
        result = _get_client().rpc("match_faq", {
            "query_embedding": embedding,   # the vector for the user's question
            "match_threshold": 0.0,         # return all results, filtering happens in Python
            "match_count": 1,               # we only need the single best match
        }).execute()

        # If a result came back, extract the FAQ id and similarity score
        if result.data:
            row = result.data[0]
            return (row["faq_id"], float(row["similarity"]))
    except Exception as e:
        print(f"[DB find_best_faq_match_db ERROR] {e}")

    # Return empty result if search failed or returned nothing
    return (None, 0.0)


def find_top_faq_matches_db(embedding: list, k: int = 3) -> list[tuple[str, float]]:
    """
    Search the FAQ vectors in Supabase and return the top-k matches.

    Used by find_top_faq_matches() in core.py so the ambiguity check
    (score gap between #1 and #2) works in pgvector mode, not just numpy mode.

    Returns [(faq_id, score), ...] sorted best-first.
    Returns [] if the search fails.
    """
    try:
        result = _get_client().rpc("match_faq", {
            "query_embedding": embedding,
            "match_threshold": 0.0,
            "match_count": k,
        }).execute()
        if result.data:
            return [(row["faq_id"], float(row["similarity"])) for row in result.data]
    except Exception as e:
        print(f"[DB find_top_faq_matches_db ERROR] {e}")
    return []


def retrieve_top_chunks_db(embedding: list, top_k: int = 4) -> list[str]:
    """
    Search the calendar chunk vectors in Supabase and return the top_k most relevant chunks.

    Calls the 'match_calendar' SQL function. These chunks are the actual calendar
    event texts that get passed to the LLM as context for answering the question.
    Returns an empty list if the search fails.
    """
    try:
        # Call the match_calendar stored procedure with the query vector
        result = _get_client().rpc("match_calendar", {
            "query_embedding": embedding,   # the vector for the user's question
            "match_count": top_k,           # how many calendar events to return
        }).execute()

        # Extract just the text from each result row (not the vector)
        return [row["chunk"] for row in result.data] if result.data else []
    except Exception as e:
        print(f"[DB retrieve_top_chunks_db ERROR] {e}")
        return []
