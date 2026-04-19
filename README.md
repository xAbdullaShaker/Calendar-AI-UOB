# UOB Calendar AI

A bilingual (Arabic + English) AI assistant for the University of Bahrain academic calendar 2025/2026.
Built with a hybrid FAQ + RAG pipeline — common questions are answered instantly from a pre-written FAQ, rare or time-relative questions fall back to a streaming LLM that only sees the 4 most relevant calendar chunks.

Includes a React web UI with UOB branding and a FastAPI backend.

---

## Features

- **Bilingual** — detects Arabic vs English automatically; supports Gulf/Khaleeji dialect question variants
- **FAQ-first** — 37 FAQ entries with 650+ question variants handle ~90% of questions with zero LLM cost
- **Streaming responses** — LLM answers type out token by token via SSE; FAQ answers appear instantly
- **Date-aware routing** — 60+ date-sensitive patterns intercept questions like "is registration open?", "did I miss it?", "الحين", "لسا", "الجاي" and route them to the LLM instead of returning a static FAQ answer
- **Upcoming vs past** — bot correctly returns upcoming events, not already-past ones (e.g. asking "when are finals?" in April 2026 returns June 2026 finals, not December 2025)
- **RAG fallback** — only the top 4 relevant calendar chunks are sent to the LLM, never the full document
- **Conversation memory** — remembers last 10 turns; detects follow-up questions and expands the embed query
- **Input sanitization** — truncates at 500 chars, strips control characters, rejects gibberish
- **Rate limiting** — max 30 messages per 10-minute rolling window per session
- **Error handling** — graceful fallback if Cohere API is unavailable; user-friendly error messages
- **React web UI** — chat interface with UOB navy/gold theme, suggestion chips, EN/AR toggle, typing indicator, bot avatar

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                        SETUP (runs once)                         │
│                                                                  │
│  uob_faq.json                                                    │
│  (37 Q&A entries) ──→ embed_faq.py ──→ Cohere Embed API         │
│                                               ↓                  │
│                                      faq_embeddings.json         │
│                                                                  │
│  uob_calendar.md                                                 │
│  (73 event chunks) ──→ embed_calendar.py ──→ Cohere Embed API   │
│                                               ↓                  │
│                                      calendar_embeddings.json    │
└──────────────────────────────────────────────────────────────────┘

                         User sends a question
                                  │
                                  ↓
                    ┌─────────────────────────┐
                    │  sanitize_input()        │
                    │  · truncate > 500 chars  │
                    │  · strip control chars   │
                    │  · reject gibberish      │
                    └────────────┬────────────┘
                                 │
                                 ↓
                    ┌─────────────────────────┐
                    │  RateLimiter             │
                    │  max 30 msgs / 10 min    │
                    │  per session             │
                    └────────────┬────────────┘
                                 │
                                 ↓
                    ┌─────────────────────────┐
                    │  is_followup()?          │
                    │  Yes → prepend previous  │
                    │  question to embed query │
                    └────────────┬────────────┘
                                 │
                                 ↓
                    ┌─────────────────────────┐
                    │  Cohere Embed API        │
                    │  query → vector          │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
               score >= 0.55            score < 0.55
                    │                         │
                    ↓                         │
        ┌───────────────────────┐             │
        │  is_date_sensitive()?  │             │
        │  60+ patterns including│             │
        │  · "did i miss"        │             │
        │  · "is it open"        │             │
        │  · "when are finals"   │             │
        │  · "when are results"  │             │
        │  · "when is eid"       │             │
        │  · "الحين" "لسا"       │             │
        │  · "الجاي" "خلص"      │             │
        └──────┬────────────────┘             │
          No   │        Yes                   │
          │    │         └────────────────────┤
          ↓    │                              ↓
  ┌────────────────┐          ┌──────────────────────────┐
  │ Pre-written FAQ│          │  Rank 73 calendar chunks  │
  │ answer returned│          │  Send top 4 to LLM with:  │
  │ instantly      │          │  · today's date + period  │
  │ AR → answer_ar │          │  · conversation history   │
  │ EN → answer_en │          │  · language instruction   │
  └────────────────┘          │  Stream tokens via SSE ──→│
                              └──────────────────────────┘
```

---

## Date-Sensitive Routing

FAQ answers are static strings — they have no knowledge of today's date.
Without `is_date_sensitive()`, a question like *"when are the final exams?"* in April 2026 would match the `fall_finals` FAQ entry and return December 2025 dates — already past.

The function checks for 60+ patterns across three categories:

| Category | Examples |
|---|---|
| Status checks | "is it open", "can i still", "did I miss", "is it too late" |
| Generic multi-semester | "when are finals", "when are results", "when is eid" |
| Gulf Arabic colloquial | "الحين", "لسا", "الجاي", "خلص", "باقي كم", "فاتني", "هالفترة" |

When triggered, the question is routed to the LLM which receives today's date and current academic period and can reason: *"finals are upcoming in June 2026"* rather than returning a hardcoded past answer.

---

## Date Awareness

Every LLM call has a date context block injected into the message:

```
Today is: Wednesday, 16 April 2026
Current academic period: Second Semester 2025/2026 (classes in progress)

CRITICAL RULES:
- You KNOW today's date. Never say "if today is..." — state facts directly.
- NEVER mention today's date in your response. Use natural relative language:
  'upcoming', 'already past', 'opens in X days', 'deadline has passed'.
```

The LLM answers relative to today without ever echoing the date back to the user.

### Academic Period Boundaries

| Period | Start | End |
|--------|-------|-----|
| First Semester (classes) | 7 Sep 2025 | 18 Dec 2025 |
| First Semester (finals) | 19 Dec 2025 | 8 Jan 2026 |
| Second Semester (classes) | 3 Feb 2026 | 14 May 2026 |
| Second Semester (finals) | 15 May 2026 | 30 May 2026 |
| Summer Session (classes) | 1 Jul 2026 | 7 Aug 2026 |
| Summer Session (finals) | 8 Aug 2026 | 14 Aug 2026 |

---

## Streaming

The `/chat/stream` endpoint returns a Server-Sent Events (SSE) stream:

```
data: {"type": "token", "text": "الامتحانات"}
data: {"type": "token", "text": " النهائية"}
...
data: {"type": "done", "source": "RAG (date-sensitive) — 73%", "warning": null}
```

- **LLM path** — tokens stream one by one as the model generates them
- **FAQ path** — full answer sent as a single `token` event immediately (no LLM latency)

The frontend fills the bot message word by word in real time.

---

## Why Not Pure RAG?

| | Pure RAG | This Project |
|---|---|---|
| Every question uses LLM | Yes | No — FAQ handles ~90% |
| LLM context | Full document | Top 4 chunks only |
| Same question = same answer | LLM may rephrase | Always identical (FAQ) |
| Hallucination risk on dates | Yes | No for FAQ; minimized for RAG |
| Cost per common question | ~$0.001 | ~$0.000001 (embed only) |
| Response speed | 1-3s always | Instant for FAQ hits |

---

## Files

| File | What it does |
|------|-------------|
| `api.py` | FastAPI backend — FAQ match, streaming SSE endpoint, RAG fallback, rate limiting |
| `chat.py` | CLI chatbot — same logic as api.py for terminal use |
| `uob_faq.json` | 37 Q&A entries with 650+ Arabic + English question variants |
| `uob_calendar.md` | Full academic calendar source document |
| `embed_faq.py` | Embeds FAQ questions — run once, re-run after editing `uob_faq.json` |
| `embed_calendar.py` | Chunks calendar into 73 events and embeds each — run once |
| `eval_threshold.py` | Evaluates FAQ similarity threshold across test cases |
| `faq_embeddings.json` | Generated by `embed_faq.py` — gitignored, must generate locally |
| `calendar_embeddings.json` | Generated by `embed_calendar.py` — gitignored, must generate locally |
| `frontend/` | React + Vite web UI |
| `requirements.txt` | Python dependencies |
| `.env` | Cohere API key — never committed |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file:
```
COHERE_API_KEY=your_key_here
```

### 3. Generate embeddings (once)

```bash
python embed_faq.py
python embed_calendar.py
```

Re-run `embed_faq.py` any time you edit `uob_faq.json`.

### 4. Start the backend

```bash
python -m uvicorn api:app --reload --port 8001
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## Tech Stack

```
Embeddings:  Cohere embed-multilingual-v3.0
LLM:         Cohere command-r-plus-08-2024
Streaming:   Server-Sent Events (SSE) via FastAPI StreamingResponse
Backend:     FastAPI + uvicorn
Frontend:    React + Vite
Matching:    Cosine similarity (numpy) on flat JSON
RAG:         Top-4 chunk retrieval per query
Language:    Arabic/Latin character ratio (>50% Arabic → AR)
Storage:     Flat JSON — no database needed
```
