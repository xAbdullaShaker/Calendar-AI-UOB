"""
api.py — FastAPI backend for UOB Calendar AI.

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

from core import (
    client,
    USE_DB,
    SIMILARITY_THRESHOLD, MAX_HISTORY,
    sanitize_input, is_arabic, build_embed_query,
    find_best_faq_match, retrieve_top_chunks,
    build_faq_response, is_date_sensitive,
    ask_llm, ask_llm_stream,
    load_faq_answers,
)

load_dotenv()


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_embeddings():
    with open("faq_embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["faqs"] if "faqs" in data else data


def load_calendar_chunks():
    with open("calendar_embeddings.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ── Rate limiter (per session) ────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, max_calls=30, window_seconds=600):
        self.max_calls = max_calls
        self.window = window_seconds
        self._store: dict[str, list[float]] = {}

    def is_allowed(self, session_id: str) -> bool:
        now = time.monotonic()
        timestamps = [t for t in self._store.get(session_id, []) if now - t < self.window]
        if len(timestamps) >= self.max_calls:
            self._store[session_id] = timestamps
            return False
        timestamps.append(now)
        self._store[session_id] = timestamps
        return True

    def seconds_until_reset(self, session_id: str) -> int:
        timestamps = self._store.get(session_id, [])
        if not timestamps:
            return 0
        return max(0, int(self.window - (time.monotonic() - timestamps[0])) + 1)


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="UOB Calendar AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

faq_answers = load_faq_answers()
rate_limiter = RateLimiter()

if USE_DB:
    faq_embeddings = None
    calendar_chunks = None
    print("pgvector mode: embeddings served from PostgreSQL")
else:
    faq_embeddings = load_embeddings()
    calendar_chunks = load_calendar_chunks()
    print(f"numpy mode: {len(faq_embeddings)} FAQ entries, {len(calendar_chunks)} calendar chunks loaded")

# In-memory session histories: { session_id: [turns] }
sessions: dict[str, list] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    source: str
    confidence: int
    warning: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_embed(text: str):
    try:
        resp = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return resp.data[0].embedding
    except Exception:
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if not rate_limiter.is_allowed(req.session_id):
        wait = rate_limiter.seconds_until_reset(req.session_id)
        raise HTTPException(status_code=429, detail=f"Too many requests. Wait {wait}s.")

    clean, warning = sanitize_input(req.message)
    if clean is None:
        raise HTTPException(status_code=400, detail=warning)

    history = sessions.get(req.session_id, [])
    arabic = is_arabic(clean)
    embed_query = build_embed_query(clean, history)

    question_embedding = get_embed(embed_query)
    if question_embedding is None:
        raise HTTPException(status_code=503, detail="AI service unavailable. Please try again.")

    best_entry, score = find_best_faq_match(question_embedding, faq_embeddings)

    def generate():
        answer_text = ""

        if score >= SIMILARITY_THRESHOLD and not is_date_sensitive(clean):
            result = build_faq_response(best_entry, arabic, score, faq_answers)
            answer_text = result["response"]
            source = f"FAQ match: {score:.0%}"
            yield f"data: {json.dumps({'type': 'token', 'text': answer_text})}\n\n"
        else:
            top_chunks = retrieve_top_chunks(question_embedding, calendar_chunks)
            source = (
                f"RAG (date-sensitive) — {score:.0%}"
                if score >= SIMILARITY_THRESHOLD
                else f"RAG fallback — {score:.0%}"
            )
            for token in ask_llm_stream(clean, top_chunks, history, arabic=arabic):
                answer_text += token
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        hist = sessions.get(req.session_id, [])
        hist.append({"question": clean, "answer": answer_text})
        if len(hist) > MAX_HISTORY:
            hist.pop(0)
        sessions[req.session_id] = hist

        yield f"data: {json.dumps({'type': 'done', 'source': source, 'warning': warning})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/clear")
def clear_history(req: ChatRequest):
    sessions.pop(req.session_id, None)
    return {"status": "cleared"}
