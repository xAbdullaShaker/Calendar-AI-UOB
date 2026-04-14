# Calendar AI — UOB

A hybrid FAQ + RAG chatbot for the University of Bahrain academic calendar 2025/2026.
Answers in Arabic and English. FAQ handles common questions instantly. RAG retrieval is used only as a fallback — the LLM never receives the full calendar.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                        SETUP (runs once)                         │
│                                                                  │
│  uob_faq.json                                                    │
│  (37 Q&A entries) ──→ embed_faq.py ──→ Cohere Embed API         │
│                                               │                  │
│                                               ↓                  │
│                                      faq_embeddings.json         │
│                                                                  │
│  uob_calendar.md                                                 │
│  (73 event chunks) ──→ embed_calendar.py ──→ Cohere Embed API   │
│                                               │                  │
│                                               ↓                  │
│                                      calendar_embeddings.json    │
└──────────────────────────────────────────────────────────────────┘

                              │
                              │ app starts
                              ↓

┌──────────────────────────────────────────────────────────────────┐
│                        RUNTIME (chat.py)                         │
│                                                                  │
│  1. Load faq_embeddings.json into memory                         │
│  2. Load calendar_embeddings.json into memory                    │
│                                                                  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ↓

                  User types a question
                 "when is registration?"
                           │
                           ↓
              ┌────────────────────────┐
              │  Detect language       │
              │  Arabic chars? → AR    │
              │  Otherwise    → EN     │
              └────────────┬───────────┘
                           │
                           ↓
              ┌────────────────────────┐
              │  Cohere Embed API      │
              │  question → vector     │
              │  [0.023, -0.45, ...]   │
              └────────────┬───────────┘
                           │
                           ↓
              ┌────────────────────────┐
              │  Compare against all   │
              │  37 FAQ vectors        │
              │  (cosine similarity)   │
              │                        │
              │  fall_drop_add: 0.81 ← │ best match
              └────────────┬───────────┘
                           │
               ┌───────────┴───────────┐
               │                       │
         score >= 0.65           score < 0.65
               │                       │
               ↓                       ↓
   ┌───────────────────┐   ┌───────────────────────────┐
   │  Return pre-      │   │  RAG: rank all 73 calendar │
   │  written answer   │   │  chunks by similarity to   │
   │  from FAQ         │   │  user question             │
   │                   │   │                            │
   │  AR → answer_ar   │   │  Take top 4 chunks only    │
   │  EN → answer_en   │   └──────────────┬─────────────┘
   └─────────┬─────────┘                  │
             │                            ↓
             │              ┌─────────────────────────┐
             │              │  Send to Cohere LLM     │
             │              │  context = top 4 chunks │
             │              │  (NOT full calendar)    │
             │              └──────────────┬──────────┘
             │                             │
             │                             ↓
             │              ┌──────────────────────────┐
             │              │  LLM reads chunks        │
             │              │  returns structured JSON │
             │              └──────────────┬───────────┘
             │                             │
             └──────────────┬──────────────┘
                            │
                            ↓
              ┌─────────────────────────────┐
              │  {                          │
              │    "ai_interpretation": "", │
              │    "response_confidence": 9,│
              │    "response": "..."        │
              │  }                          │
              │  [FAQ match: 81%]           │
              │  [RAG fallback: 51%, 4 chks]│
              └─────────────────────────────┘
```

---

## Why Not Pure RAG?

| | Pure RAG | This Project |
|---|---|---|
| Every question uses LLM | Yes | No — FAQ answers ~90% |
| LLM context | Full document | Top 4 retrieved chunks only |
| Same question = same answer | LLM may rephrase | Always identical (FAQ) |
| Hallucination risk on dates | Yes | No for FAQ; minimized for RAG |
| Cost per common question | ~$0.001 | $0.00 |

For common questions: FAQ returns a pre-written answer instantly, no LLM involved.
For edge cases: RAG retrieves only the relevant chunks — the LLM never sees the full calendar.

---

## What Embeddings Do

An embedding converts a sentence into numbers that represent its **meaning** — not its words.

```
"When do classes start?"       -> [0.023, -0.451, 0.782, ...]
"When is my first day at uni?" -> [0.025, -0.447, 0.780, ...]  <- close
"متى تبدأ الدراسة؟"            -> [0.021, -0.449, 0.779, ...]  <- also close
```

Arabic and English questions match each other automatically — no translation needed.
The same embedding is used for both FAQ matching and calendar chunk retrieval.

---

## Files

| File | What it does |
|------|-------------|
| `uob_faq.json` | 37 Q&A entries with Arabic + English phrasings and pre-written answers |
| `uob_calendar.md` | Full academic calendar — chunked and embedded for RAG retrieval |
| `embed_faq.py` | Converts FAQ questions to vectors — run once |
| `embed_calendar.py` | Chunks uob_calendar.md into 73 events and embeds each one — run once |
| `chat.py` | The bot — FAQ match first, RAG retrieval on fallback |
| `faq_embeddings.json` | Generated by embed_faq.py — stores FAQ vectors locally |
| `calendar_embeddings.json` | Generated by embed_calendar.py — stores 73 calendar chunk vectors |
| `requirements.txt` | Python dependencies |
| `.env` | Your Cohere API key — never pushed to GitHub |
| `.env.example` | Empty template for the .env file |
| `docs/architecture.md` | Full technical architecture detail |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Cohere API key
cp .env.example .env
# edit .env and paste your key

# 3. Convert FAQ to embeddings (once)
python embed_faq.py

# 4. Chunk and embed the calendar (once)
python embed_calendar.py

# 5. Start chatting
python chat.py
```

Re-run `embed_faq.py` if you update `uob_faq.json`.
Re-run `embed_calendar.py` if you update `uob_calendar.md`.

---

## Tech Stack

```
Embeddings:  Cohere embed-multilingual-v3.0
LLM:         Cohere command-a-03-2025
Matching:    Cosine similarity on flat JSON
RAG:         Top-4 chunk retrieval per query
Language:    Detected by Arabic Unicode range check
Storage:     Flat JSON files — no database needed
```
