# Calendar AI — UOB (University of Business)

An AI chatbot that answers questions about the UOB academic calendar. Designed to be fast and cost-efficient by answering most questions from a local FAQ, and only falling back to an LLM for unusual questions.

---

## What This Project Does

- Converts the UOB academic calendar PDF into AI-readable formats
- Provides a structured FAQ in JSON for instant answers
- Uses **semantic embeddings** so the bot understands meaning — not just exact words
- Answers in both **Arabic and English**
- Restricts the AI to only answer from the calendar data

---

## Files in This Repo

| File | Purpose |
|------|---------|
| `uob_calendar.md` | Clean Markdown tables of all academic dates (semesters, holidays, exams) |
| `uob_calendar.json` | Same data in structured JSON — bilingual (English + Arabic labels) |
| `uob_faq.json` | FAQ with multiple question phrasings per answer (Arabic + English) |
| `uob_ai_system_prompt.md` | Ready-to-use system prompt that restricts the AI to calendar data only |
| `docs/architecture.md` | How the system works (FAQ matching, embeddings, LLM fallback) |

---

## Architecture

### The Problem
Users ask the same question in many different ways:
- "When does spring start?"
- "First day of second semester?"
- "متى تبدأ الدراسة؟"

A simple keyword match fails for synonyms and different languages.

### The Solution: 3-Layer Approach

```
User Question
      ↓
Layer 1: Exact / Keyword Match  →  Answer (FREE, instant, ~0ms)
      ↓ no match
Layer 2: Semantic Embedding Match  →  Answer (near-free, ~50ms)
      ↓ similarity < 75%
Layer 3: LLM with calendar context  →  Answer (costs tokens)
```

Layer 1 and 2 handle ~90% of questions. The LLM is only called for edge cases.

---

## How Embeddings Work

An **embedding** converts a sentence into an array of ~1536 numbers that represent its *meaning*.

Sentences with similar meanings produce similar numbers — even across languages.

```
"When do classes start?"       → [0.023, -0.451, 0.782, ...]
"متى تبدأ الدراسة؟"            → [0.021, -0.449, 0.779, ...]  ← very close!
"When is my first day at uni?" → [0.025, -0.447, 0.780, ...]  ← also close
```

### Two-Step Process

**Step 1 — Setup (runs once):**
```
Read uob_faq.json
    ↓
Send each question to embedding API (OpenAI / Cohere)
    ↓
Get back a vector (array of numbers)
    ↓
Save to faq_embeddings.json
```

**Step 2 — Runtime (every user message):**
```
User's question
    ↓
Convert to vector
    ↓
Compare (cosine similarity) against all saved vectors
    ↓
If best match > 75% → return FAQ answer (FREE)
If below 75%        → fallback to LLM
```

---

## FAQ Data Format

Each FAQ entry in `uob_faq.json` has:

```json
{
  "id": "fall_start",
  "tags": ["semester", "start"],
  "questions": [
    "when does the first semester start",
    "first day of fall semester",
    "متى يبدأ الفصل الأول",
    "أول يوم دراسة"
  ],
  "answer_en": "Classes begin Sunday, 7 September 2025.",
  "answer_ar": "تبدأ الدراسة يوم الأحد 7 سبتمبر 2025."
}
```

Multiple question phrasings per entry = better matching without needing the LLM.

---

## Embedding Tools (Recommended)

| Tool | Cost | Arabic Support | Notes |
|------|------|---------------|-------|
| OpenAI `text-embedding-3-small` | ~$0.02 / 1M tokens | Excellent | Easiest setup |
| Cohere `embed-multilingual` | Free tier available | Excellent | Best for Arabic specifically |
| Sentence Transformers (local) | FREE | Good | Runs on your server, no API needed |

For ~26 FAQ entries, converting all questions costs **less than $0.001 total**.

---

## How to Use the System Prompt

The file `uob_ai_system_prompt.md` is a ready-to-use prompt that:
- Restricts the AI to only answer from the calendar data
- Handles both Arabic and English
- Refuses off-topic questions
- Notes that moon-sighting-dependent holidays may shift by a day (marked with `*`)

### Quickest Setup (Claude Projects / ChatGPT GPTs)
1. Create a new Project or GPT
2. Paste the system prompt as the instructions
3. Upload `uob_calendar.md` as the knowledge file

### Via API
```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    system=open("uob_ai_system_prompt.md").read() + "\n\n" + open("uob_calendar.md").read(),
    messages=[{"role": "user", "content": user_question}]
)
```

---

## Decision Guide: Which Data Format to Use?

| Use Case | Best Format |
|----------|------------|
| Chat-only AI (your case) | `uob_calendar.md` — LLMs read Markdown tables very well |
| App with filters/search | `uob_calendar.json` — programmatic queries |
| FAQ matching bot | `uob_faq.json` + embeddings |
| All of the above | JSON as source of truth, generate Markdown from it |

---

## Notes on Dates

Dates marked with `*` are **Hijri-based** and depend on moon sighting. They may shift by ±1 day. The system prompt already instructs the AI to mention this when relevant.

---

## Tech Stack Suggestion

```
Frontend:   Any chat UI (web, mobile, WhatsApp bot)
Matching:   Python (rapidfuzz) or JS (fuse.js) for keyword match
Embeddings: OpenAI or Cohere API
Fallback:   Claude API (Anthropic) with uob_calendar.md as context
Storage:    faq_embeddings.json (flat file, no DB needed at this scale)
```
