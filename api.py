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
    SIMILARITY_THRESHOLD, AMBIGUITY_GAP, MAX_HISTORY,
    sanitize_input, is_arabic, build_embed_query,
    find_top_faq_matches, retrieve_top_chunks,
    faq_domain_matches, build_faq_response, is_date_sensitive,
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

EMBED_MODEL = "text-embedding-3-large"


def get_embed(text: str):
    try:
        resp = client.embeddings.create(input=[text], model=EMBED_MODEL)
        return resp.data[0].embedding
    except Exception as e:
        print(f"[get_embed ERROR] {e}")
        return None


def log_query(question: str, faq_id: str, score: float, source: str, answer: str):
    """Append one row to query_log.csv for post-hoc quality review."""
    import csv, datetime
    row = [
        datetime.datetime.now().isoformat(),
        question,
        faq_id,
        f"{score:.2%}",
        source,
        answer[:300].replace("\n", " "),
    ]
    try:
        with open("query_log.csv", "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        print(f"[log_query ERROR] {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    print(f"[REQUEST] message={req.message!r}")
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

    top_matches = find_top_faq_matches(question_embedding, faq_embeddings)
    best_entry, score = top_matches[0] if top_matches else (None, 0.0)
    second_score = top_matches[1][1] if len(top_matches) > 1 else 0.0
    ambiguous = (score - second_score) < AMBIGUITY_GAP

    def generate():
        answer_text = ""
        source = f"RAG fallback — {score:.0%}"
        try:
            date_sensitive = is_date_sensitive(embed_query)   # run on normalized query
            domain_ok = faq_domain_matches(clean, best_entry["id"]) if best_entry else False

            print(
                f"[FAQ] best={best_entry}, score={score:.0%}, second={second_score:.0%}, "
                f"ambiguous={ambiguous}, date_sensitive={date_sensitive}, domain_ok={domain_ok}"
            )

            if (score >= SIMILARITY_THRESHOLD
                    and not date_sensitive
                    and not ambiguous
                    and domain_ok):
                result = build_faq_response(best_entry, arabic, score, faq_answers)
                answer_text = result["response"]
                source = f"FAQ match: {score:.0%}"
                yield f"data: {json.dumps({'type': 'token', 'text': answer_text})}\n\n"
            else:
                top_chunks = retrieve_top_chunks(question_embedding, calendar_chunks)
                if date_sensitive:
                    source = f"RAG (date-sensitive) — {score:.0%}"
                elif ambiguous:
                    source = f"RAG (ambiguous match) — {score:.0%}"
                elif not domain_ok:
                    source = f"RAG (domain mismatch) — {score:.0%}"
                else:
                    source = f"RAG fallback — {score:.0%}"
                print(f"[RAG] chunks={len(top_chunks)}, source={source}")
                for token in ask_llm_stream(clean, top_chunks, history, arabic=arabic):
                    answer_text += token
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            log_query(clean, best_entry["id"] if best_entry else "none", score, source, answer_text)

            hist = sessions.get(req.session_id, [])
            hist.append({"question": clean, "answer": answer_text})
            if len(hist) > MAX_HISTORY:
                hist.pop(0)
            sessions[req.session_id] = hist
        except Exception as e:
            print(f"[generate ERROR] {e}")
            if not answer_text:
                err = "عذراً، حدث خطأ. يرجى المحاولة مجدداً." if arabic else "Something went wrong. Please try again."
                yield f"data: {json.dumps({'type': 'token', 'text': err})}\n\n"

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
