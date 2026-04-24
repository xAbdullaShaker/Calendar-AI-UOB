"""
api.py — FastAPI backend for UOB Calendar AI.

This is the web server. It listens for chat messages from the frontend,
runs the full FAQ + RAG pipeline, and streams the answer back token by token.

Every request goes through this sequence:
  sanitize → rate limit → normalize → embed → FAQ match → route → respond → log

Usage:
    python -m uvicorn api:app --reload --port 8001
"""

import json
import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import all shared logic from core.py
from core import (
    client,
    USE_DB,
    SIMILARITY_THRESHOLD, AMBIGUITY_GAP, MAX_HISTORY,
    sanitize_input, is_arabic, build_embed_query,
    find_top_faq_matches, retrieve_top_chunks,
    faq_domain_matches, build_faq_response, is_date_sensitive,
    ask_llm, ask_llm_stream,
    load_faq_answers, GREETINGS,
)

# Load environment variables from .env (OPENAI_API_KEY, SUPABASE_URL, etc.)
load_dotenv()


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_embeddings():
    """
    Load FAQ embeddings from the local JSON file.
    Only used in numpy mode (when SUPABASE_URL is not set).
    """
    with open("faq_embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["faqs"] if "faqs" in data else data


def load_calendar_chunks():
    """
    Load calendar chunk embeddings from the local JSON file.
    Only used in numpy mode (when SUPABASE_URL is not set).
    """
    with open("calendar_embeddings.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ── Rate limiter (per session) ────────────────────────────────────────────────

class RateLimiter:
    """
    Tracks message counts per session and enforces a rolling rate limit.
    Each session gets max_calls messages within any window_seconds window.
    Uses a monotonic clock so it's not affected by system time changes.
    """
    def __init__(self, max_calls=30, window_seconds=600):
        self.max_calls = max_calls         # max messages per window (30)
        self.window = window_seconds        # window size in seconds (600 = 10 min)
        self._store: dict[str, list[float]] = {}  # session_id → list of timestamps

    def is_allowed(self, session_id: str) -> bool:
        """
        Return True if this session can send another message.
        Removes old timestamps outside the window before checking.
        """
        now = time.monotonic()
        # Keep only timestamps within the rolling window
        timestamps = [t for t in self._store.get(session_id, []) if now - t < self.window]
        if len(timestamps) >= self.max_calls:
            # Too many recent messages — update the store and deny
            self._store[session_id] = timestamps
            return False
        # Record this new message timestamp and allow
        timestamps.append(now)
        self._store[session_id] = timestamps
        return True

    def seconds_until_reset(self, session_id: str) -> int:
        """
        Return how many seconds until the oldest message in the window expires.
        Used to tell the user how long they need to wait.
        """
        timestamps = self._store.get(session_id, [])
        if not timestamps:
            return 0
        return max(0, int(self.window - (time.monotonic() - timestamps[0])) + 1)


# ── App setup ─────────────────────────────────────────────────────────────────

# Create the FastAPI application
app = FastAPI(title="UOB Calendar AI")

# Allow requests from any origin (needed for the React frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAQ answer text at startup (used to look up answers by FAQ id)
faq_answers = load_faq_answers()

# Create rate limiter instance (shared across all requests)
rate_limiter = RateLimiter()


def _validate_db() -> bool:
    """
    Test the DB vector search with a dummy query to confirm dimensions match.

    We probe both match_faq and match_calendar with a zero vector of the correct
    embedding dimension. If either call returns a dimension-mismatch error (Postgres
    code 22000) it means the DB was populated with a different model — fall back to
    numpy so the server still works correctly.

    Returns True if the DB is healthy, False if we should fall back to numpy.
    """
    try:
        from db import _get_client
        dummy = [0.0] * 3072  # must match EMBED_MODEL (text-embedding-3-large)
        _get_client().rpc("match_faq", {
            "query_embedding": dummy,
            "match_threshold": 0.0,
            "match_count": 1,
        }).execute()
        _get_client().rpc("match_calendar", {
            "query_embedding": dummy,
            "match_count": 1,
        }).execute()
        return True
    except Exception as e:
        err = str(e)
        if "different vector dimensions" in err:
            print(
                "[STARTUP] DB vector dimension mismatch — the DB was populated with a "
                "different embedding model. Falling back to numpy (local JSON files).\n"
                "[STARTUP] To fix: re-run migrate_to_pgvector.py after clearing the DB "
                "tables and updating the SQL RPC functions to use vector(3072)."
            )
        else:
            print(f"[STARTUP] DB health check failed: {e}. Falling back to numpy.")
        return False


# Load embeddings into memory or use database depending on environment.
# If the DB health check fails (e.g. vector dimension mismatch from a stale migration),
# override core.USE_DB so all retrieval functions fall back to numpy automatically.
import core as _core

if USE_DB:
    if _validate_db():
        # pgvector mode — embeddings stay in PostgreSQL, no local files needed
        faq_embeddings = None
        calendar_chunks = None
        print("pgvector mode: embeddings served from PostgreSQL")
    else:
        # DB unhealthy — force numpy mode for all retrieval functions
        _core.USE_DB = False
        faq_embeddings = load_embeddings()
        calendar_chunks = load_calendar_chunks()
        print(f"numpy fallback mode: {len(faq_embeddings)} FAQ entries, {len(calendar_chunks)} calendar chunks loaded")
else:
    # numpy mode — SUPABASE_URL not set
    faq_embeddings = load_embeddings()
    calendar_chunks = load_calendar_chunks()
    print(f"numpy mode: {len(faq_embeddings)} FAQ entries, {len(calendar_chunks)} calendar chunks loaded")

# In-memory conversation histories: { session_id: [{"question": ..., "answer": ...}] }
sessions: dict[str, list] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """The JSON body sent by the frontend for each chat message."""
    message: str
    session_id: str = "default"  # identifies the user's browser session


class ChatResponse(BaseModel):
    """Response format for non-streaming endpoint (not currently used in UI)."""
    response: str
    source: str
    confidence: int
    warning: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

# The embedding model used for live query embedding — must match what embed_faq.py used
EMBED_MODEL = "text-embedding-3-large"


def get_embed(text: str):
    """
    Convert a text string into an embedding vector using OpenAI.
    Returns the vector (list of floats) or None if the API call fails.
    """
    try:
        resp = client.embeddings.create(input=[text], model=EMBED_MODEL)
        return resp.data[0].embedding
    except Exception as e:
        print(f"[get_embed ERROR] {e}")
        return None


def log_query(question: str, faq_id: str, score: float, source: str, answer: str):
    """
    Append one row to query_log.csv for post-hoc quality review.

    The log lets you review which questions are routing incorrectly,
    which FAQ entries are getting wrong matches, and what answers were returned.
    Each row has: timestamp, question, matched_faq_id, score, source, answer_excerpt.
    """
    import csv, datetime
    row = [
        datetime.datetime.now().isoformat(),  # when the question was asked
        question,                              # the user's original question
        faq_id,                               # which FAQ entry matched (or "none")
        f"{score:.2%}",                       # similarity score as percentage
        source,                               # how the answer was generated
        answer[:300].replace("\n", " "),      # first 300 chars of the answer
    ]
    try:
        with open("query_log.csv", "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        print(f"[log_query ERROR] {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Main chat endpoint — handles one user message and streams the answer back.

    Uses Server-Sent Events (SSE) to send tokens one at a time so the UI can
    display them as they arrive (typewriter effect).

    The pipeline:
    1. Rate limit check
    2. Input sanitization
    3. Build normalized embed query (follow-up expansion + dialect + spell)
    4. Embed the query
    5. Find top-3 FAQ matches
    6. Route: FAQ (if all 4 checks pass) or RAG (if any check fails)
    7. Save answer to session history
    8. Log to query_log.csv
    9. Send 'done' event with source label
    """
    print(f"[REQUEST] message={req.message!r}")

    # ── Step 1: Rate limit check ──────────────────────────────────────────────
    if not rate_limiter.is_allowed(req.session_id):
        wait = rate_limiter.seconds_until_reset(req.session_id)
        raise HTTPException(status_code=429, detail=f"Too many requests. Wait {wait}s.")

    # ── Step 2: Sanitize and validate input ───────────────────────────────────
    clean, warning = sanitize_input(req.message)
    if clean is None:
        # Input was rejected entirely (gibberish, symbols only, etc.)
        raise HTTPException(status_code=400, detail=warning)

    # ── Step 2.5: Short-circuit greetings before the embedding pipeline ─────────
    arabic = is_arabic(clean)
    normalized_clean = clean.strip().lower()
    if normalized_clean in GREETINGS:
        greeting_entry = {"id": "greeting"}
        result = build_faq_response(greeting_entry, arabic, 1.0, faq_answers)

        def greeting_stream():
            yield f"data: {json.dumps({'type': 'token', 'text': result['response']})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'source': 'FAQ match: 100%', 'warning': warning})}\n\n"

        return StreamingResponse(
            greeting_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Step 3: Load session history and build embed query ────────────────────
    history = sessions.get(req.session_id, [])

    # Build the normalized query: expands follow-ups, maps dialect, corrects typos
    embed_query = build_embed_query(clean, history)

    # ── Step 4: Embed the normalized query ────────────────────────────────────
    question_embedding = get_embed(embed_query)
    if question_embedding is None:
        raise HTTPException(status_code=503, detail="AI service unavailable. Please try again.")

    # ── Step 5: Find top-3 FAQ candidates by cosine similarity ───────────────
    top_matches = find_top_faq_matches(question_embedding, faq_embeddings)
    best_entry, score = top_matches[0] if top_matches else (None, 0.0)
    second_score = top_matches[1][1] if len(top_matches) > 1 else 0.0

    # Ambiguity check: if top two scores are very close, we're not confident enough
    ambiguous = (score - second_score) < AMBIGUITY_GAP

    def generate():
        """
        Generator function that yields SSE events.
        Each 'token' event carries a piece of the answer text.
        The final 'done' event carries the source label and any warning.
        """
        answer_text = ""
        source = f"RAG fallback — {score:.0%}"  # default; overwritten below if FAQ is used
        try:
            # ── Step 6: Routing — decide FAQ or RAG ──────────────────────────

            # Check if the question needs today's date to answer correctly
            # Run on embed_query (normalized) to catch dialect date phrases
            date_sensitive = is_date_sensitive(embed_query)

            # Check if the matched FAQ's topic matches what the question is about
            domain_ok = faq_domain_matches(clean, best_entry["id"]) if best_entry else False

            print(
                f"[FAQ] best={best_entry['id'] if best_entry else None}, score={score:.0%}, "
                f"second={second_score:.0%}, ambiguous={ambiguous}, "
                f"date_sensitive={date_sensitive}, domain_ok={domain_ok}"
            )

            if (score >= SIMILARITY_THRESHOLD   # score is high enough
                    and not date_sensitive       # doesn't need today's date
                    and not ambiguous            # top match is clearly better than #2
                    and domain_ok):             # question topic matches the FAQ topic
                # ── FAQ path: all checks passed, return pre-written answer ───
                result = build_faq_response(best_entry, arabic, score, faq_answers)
                answer_text = result["response"]
                source = f"FAQ match: {score:.0%}"
                # Send the full answer as a single token event (instant, no streaming)
                yield f"data: {json.dumps({'type': 'token', 'text': answer_text})}\n\n"

            else:
                # ── RAG path: find most relevant calendar chunks, ask the LLM ─
                top_chunks = retrieve_top_chunks(question_embedding, calendar_chunks)

                # Label explains exactly why we went to RAG
                if date_sensitive:
                    source = f"RAG (date-sensitive) — {score:.0%}"
                elif ambiguous:
                    source = f"RAG (ambiguous match) — {score:.0%}"
                elif not domain_ok:
                    source = f"RAG (domain mismatch) — {score:.0%}"
                else:
                    source = f"RAG fallback — {score:.0%}"

                print(f"[RAG] chunks={len(top_chunks)}, source={source}")

                # Stream the LLM answer token by token to the frontend
                for token in ask_llm_stream(clean, top_chunks, history, arabic=arabic):
                    answer_text += token
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            # ── Step 8: Log the query ─────────────────────────────────────────
            log_query(clean, best_entry["id"] if best_entry else "none", score, source, answer_text)

            # ── Step 7: Save this turn to session history ─────────────────────
            hist = sessions.get(req.session_id, [])
            hist.append({"question": clean, "answer": answer_text})
            # Keep only the most recent MAX_HISTORY turns
            if len(hist) > MAX_HISTORY:
                hist.pop(0)
            sessions[req.session_id] = hist

        except Exception as e:
            print(f"[generate ERROR] {e}")
            # If something went wrong and no answer was sent yet, send an error message
            if not answer_text:
                err = "عذراً، حدث خطأ. يرجى المحاولة مجدداً." if arabic else "Something went wrong. Please try again."
                yield f"data: {json.dumps({'type': 'token', 'text': err})}\n\n"

        # ── Step 9: Send the final 'done' event with metadata ─────────────────
        # The frontend uses this to stop the typing indicator and show the source tag
        yield f"data: {json.dumps({'type': 'done', 'source': source, 'warning': warning})}\n\n"

    # Return a streaming response — the browser receives events as they are yielded
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/clear")
def clear_history(req: ChatRequest):
    """
    Clear the conversation history for a session.
    Called when the user clicks the 'New Chat' button in the frontend.
    """
    sessions.pop(req.session_id, None)
    return {"status": "cleared"}
