# UOB Calendar AI

A bilingual (Arabic + English) AI assistant for the University of Bahrain academic calendar 2025/2026.
Built with a hybrid FAQ + RAG pipeline — common questions are answered instantly from a pre-written FAQ, rare or complex questions fall back to a retrieval-augmented LLM that only sees the 4 most relevant calendar chunks.

Includes a React web UI with UOB branding and a FastAPI backend.

---

## Features

- **Bilingual** — detects Arabic vs English automatically using character ratio (not just any-match)
- **FAQ-first** — 37 FAQ entries with 500+ question variants handle ~90% of questions with zero LLM cost
- **RAG fallback** — only the top 4 relevant calendar chunks are sent to the LLM, never the full document
- **Conversation memory** — remembers last 10 turns; detects follow-up questions and expands the embedding query
- **Date-aware** — injects today's date and current academic period into every LLM call so answers like "is registration open now?" and "how many days until finals?" are grounded in the actual date
- **Input sanitization** — truncates at 500 chars, strips control characters, rejects gibberish
- **Rate limiting** — max 10 messages per 60-second rolling window per session
- **React web UI** — chat interface with UOB navy/gold theme, suggestion chips, EN/AR language toggle, bot avatar

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
                    │  · rate limit check      │
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
                    ↓                         ↓
        ┌───────────────────┐   ┌─────────────────────────┐
        │  Pre-written FAQ  │   │  Rank 73 calendar chunks │
        │  answer returned  │   │  Send top 4 to LLM       │
        │  AR → answer_ar   │   │  + today's date context  │
        │  EN → answer_en   │   │  + conversation history  │
        └───────────────────┘   └─────────────────────────┘
```

---

## Date Awareness

Every LLM call receives a `Current Date Context` block that includes:
- Today's date (e.g. `Tuesday, 15 April 2026`)
- Current academic period (e.g. `Second Semester 2025/2026 — classes in progress`)
- Instructions to resolve relative questions: "is registration open now?", "how many days until finals?", "did I miss the drop deadline?"

The academic period is calculated in real-time against the actual UOB calendar boundaries — no hardcoding.

---

## Why Not Pure RAG?

| | Pure RAG | This Project |
|---|---|---|
| Every question uses LLM | Yes | No — FAQ answers ~90% |
| LLM context | Full document | Top 4 chunks only |
| Same question = same answer | LLM may rephrase | Always identical (FAQ) |
| Hallucination risk on dates | Yes | No for FAQ; minimized for RAG |
| Cost per common question | ~$0.001 | ~$0.000001 (embed only) |

---

## Files

| File | What it does |
|------|-------------|
| `chat.py` | CLI chatbot — FAQ match first, RAG on fallback |
| `api.py` | FastAPI backend — wraps chat.py logic as a REST API |
| `uob_faq.json` | 37 Q&A entries with 500+ Arabic + English question variants |
| `uob_calendar.md` | Full academic calendar source |
| `embed_faq.py` | Embeds FAQ questions — run once, re-run after editing uob_faq.json |
| `embed_calendar.py` | Chunks calendar into 73 events and embeds each — run once |
| `eval_threshold.py` | Evaluates FAQ similarity threshold across 35 test cases |
| `faq_embeddings.json` | Generated by embed_faq.py |
| `calendar_embeddings.json` | Generated by embed_calendar.py |
| `frontend/` | React + Vite web UI |
| `requirements.txt` | Python dependencies |
| `.env` | Cohere API key — never pushed to GitHub |

---

## Setup

### Backend (CLI)

```bash
pip install -r requirements.txt

# Add your Cohere API key
cp .env.example .env

# Embed FAQ and calendar (once)
python embed_faq.py
python embed_calendar.py

# Start CLI chatbot
python chat.py
```

### Web UI

```bash
# Terminal 1 — backend
python -m uvicorn api:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` (or whichever port Vite picks).

Re-run `embed_faq.py` after editing `uob_faq.json`.
Re-run `embed_calendar.py` after editing `uob_calendar.md`.

---

## Tech Stack

```
Embeddings:  Cohere embed-multilingual-v3.0
LLM:         Cohere command-a-03-2025
Backend:     FastAPI + uvicorn
Frontend:    React + Vite
Matching:    Cosine similarity on flat JSON
RAG:         Top-4 chunk retrieval per query
Language:    Arabic/Latin character ratio (>50% Arabic → AR)
Storage:     Flat JSON — no database needed
```
