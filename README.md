# Calendar AI — UOB

A hybrid FAQ chatbot for the University of Bahrain academic calendar 2025/2026.
Answers in Arabic and English. Uses semantic embeddings for smart matching — the LLM is only a fallback, not the default.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        SETUP (runs once)                        │
│                                                                 │
│  uob_faq.json                                                   │
│  (37 Q&A entries)  ──→  embed_faq.py  ──→  Cohere Embed API    │
│                                                  │              │
│                                                  ↓              │
│                                        faq_embeddings.json      │
│                                        (37 entries as vectors)  │
└─────────────────────────────────────────────────────────────────┘

                              │
                              │ app starts
                              ↓

┌─────────────────────────────────────────────────────────────────┐
│                       RUNTIME (chat.py)                         │
│                                                                 │
│  1. Load faq_embeddings.json into memory                        │
│  2. Load uob_calendar.md into memory                            │
│                                                                 │
└──────────────────────────────┬──────────────────────────────────┘
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
                  │  Send question to      │
                  │  Cohere Embed API      │
                  │                        │
                  │  "when is registration"│
                  │        ↓               │
                  │  [0.023, -0.45, ...]   │
                  └────────────┬───────────┘
                               │
                               ↓
                  ┌────────────────────────┐
                  │  Compare against all   │
                  │  37 FAQ vectors        │
                  │  (cosine similarity)   │
                  │                        │
                  │  fall_start:    0.51   │
                  │  fall_drop_add: 0.81 ← │ best match
                  │  fall_finals:   0.43   │
                  │  ...                   │
                  └────────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
              score >= 0.65         score < 0.65
                    │                     │
                    ↓                     ↓
        ┌───────────────────┐   ┌──────────────────────┐
        │  Return pre-      │   │  Send to Cohere LLM  │
        │  written answer   │   │  (command-a-03-2025) │
        │  from FAQ         │   │                      │
        │                   │   │  context:            │
        │  AR → answer_ar   │   │  uob_calendar.md     │
        │  EN → answer_en   │   │  + user question     │
        └─────────┬─────────┘   └──────────┬───────────┘
                  │                         │
                  │                         ↓
                  │             ┌───────────────────────┐
                  │             │  LLM reads calendar   │
                  │             │  generates answer     │
                  │             │  in user's language   │
                  │             └──────────┬────────────┘
                  │                        │
                  └───────────┬────────────┘
                              │
                              ↓
                  ┌───────────────────────┐
                  │   Bot replies to      │
                  │   user               │
                  │                      │
                  │  + shows source:     │
                  │  [FAQ match: 81%]    │
                  │  [LLM fallback: 51%] │
                  └───────────────────────┘
```

---

## Why Not Pure RAG?

| | Pure RAG | This Project |
|---|---|---|
| Every question uses LLM | Yes | No — FAQ first |
| LLM only for edge cases | No | Yes |
| Same question = same answer | LLM may rephrase | Always identical |
| Hallucination risk on dates | Yes | No (pre-written) |
| Cost per common question | ~$0.001 | $0.00 |

A calendar has **fixed, factual answers**. Pre-writing them is safer and cheaper than generating them every time.

---

## What Embeddings Do

An embedding converts a sentence into numbers that represent its **meaning** — not its words.

```
"When do classes start?"       -> [0.023, -0.451, 0.782, ...]
"When is my first day at uni?" -> [0.025, -0.447, 0.780, ...]  <- close
"متى تبدأ الدراسة؟"            -> [0.021, -0.449, 0.779, ...]  <- also close
```

Arabic and English questions match each other automatically — no translation needed.

---

## Files

| File | What it does |
|------|-------------|
| `uob_faq.json` | 37 Q&A entries with Arabic + English phrasings and pre-written answers |
| `uob_calendar.md` | Full academic calendar — fed to LLM as context when FAQ has no match |
| `embed_faq.py` | Converts FAQ questions to vectors via Cohere — run once |
| `chat.py` | The bot — handles user input, matching, FAQ reply or LLM fallback |
| `faq_embeddings.json` | Generated by embed_faq.py — stores all vectors locally |
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

# 4. Start chatting
python chat.py
```

---

## Tech Stack

```
Embeddings:  Cohere embed-multilingual-v3.0
LLM:         Cohere command-a-03-2025
Matching:    Cosine similarity on flat JSON
Language:    Detected by Arabic Unicode range check
Storage:     Flat JSON files — no database needed
```
