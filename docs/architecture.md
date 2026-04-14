# System Architecture — UOB Calendar AI

## Overview

The UOB Calendar AI is a cost-efficient FAQ chatbot. Most questions are answered instantly from a local JSON file. The LLM is only called when no FAQ match is found.

---

## Flow Diagram

```
┌─────────────────────────────────────────────────┐
│                  User sends question             │
└──────────────────────┬──────────────────────────┘
                       ↓
         ┌─────────────────────────┐
         │  Detect language        │
         │  (Arabic or English)    │
         └──────────┬──────────────┘
                    ↓
         ┌─────────────────────────┐
         │  Layer 1: Keyword match │ ──→ Match found → return answer_ar / answer_en
         └──────────┬──────────────┘
                    ↓ no match
         ┌─────────────────────────┐
         │  Layer 2: Embedding     │
         │  similarity search      │ ──→ Similarity > 0.75 → return answer
         └──────────┬──────────────┘
                    ↓ similarity < 0.75
         ┌─────────────────────────┐
         │  Layer 3: LLM call      │
         │  (Claude / GPT)         │
         │  with uob_calendar.md   │ ──→ return LLM answer
         └─────────────────────────┘
```

---

## Layer 1: Keyword Matching

**How it works:**
- Normalize both the user question and FAQ questions (lowercase, remove punctuation)
- Tokenize into words
- Count word overlap
- If overlap score is high enough → return answer

**Why use it:**
- Zero cost
- Zero latency
- Handles exact and near-exact matches

**Tools:**
- JavaScript: `fuse.js`
- Python: `rapidfuzz` or `thefuzz`

**Limitation:** Fails on synonyms. "When does school start" won't match "first day of semester" because the words are different.

---

## Layer 2: Semantic Embedding Search

**How it works:**

1. **One-time setup:** Each FAQ question is converted to a vector (1536 numbers) using an embedding API. Stored in `faq_embeddings.json`.

2. **At runtime:** The user's question is converted to a vector. Cosine similarity is calculated against all stored vectors. The closest match is returned if similarity > 0.75.

**Why use it:**
- Understands *meaning*, not just words
- Handles Arabic ↔ English automatically
- Works for paraphrasing, synonyms, different sentence structures

**Example:**
```
"When do classes start?"           similarity: 0.94  → MATCH
"متى تبدأ الدراسة؟"                similarity: 0.91  → MATCH
"first day at uni?"                similarity: 0.87  → MATCH
"what is the university address?"  similarity: 0.31  → NO MATCH → go to Layer 3
```

**Cost:**
- Setup: < $0.001 for all 26 FAQ entries
- Per query: < $0.000001

---

## Layer 3: LLM Fallback

Only triggered when layers 1 and 2 don't find a good match.

**System prompt instructs the LLM to:**
- Only answer from the provided calendar data
- Respond in the same language the user used
- Say "I don't have that information" if the question is outside the calendar scope
- Note that `*`-marked dates depend on moon sighting

**Context provided to LLM:**
- Full `uob_calendar.md` content

---

## Data Files

### `uob_faq.json`
The FAQ database. Each entry:
```json
{
  "id": "unique_id",
  "tags": ["category1", "category2"],
  "questions": ["English phrasing", "Arabic phrasing", "..."],
  "answer_en": "Answer in English.",
  "answer_ar": "الإجابة بالعربي."
}
```

### `faq_embeddings.json` (generated, not in repo)
Produced by the setup script. Contains vectors for all FAQ questions. Not committed to the repo since it can be regenerated.

### `uob_calendar.md`
Full academic calendar in Markdown tables. Fed to the LLM as context.

---

## Language Detection

Simple heuristic: check if the input contains Arabic Unicode characters (`\u0600`–`\u06FF`). If yes → respond with `answer_ar`. If no → respond with `answer_en`.

```python
def is_arabic(text):
    return any('\u0600' <= c <= '\u06FF' for c in text)
```

---

## Scaling Notes

At 26 FAQ entries, no database is needed — flat JSON files are sufficient.

If the FAQ grows to 100+ entries:
- Use a vector database: **Pinecone**, **Chroma**, or **Supabase pgvector**
- This keeps similarity search fast at scale

---

## Recommended Stack

| Component | Tool |
|-----------|------|
| Embeddings | OpenAI `text-embedding-3-small` or Cohere `embed-multilingual` |
| Keyword matching | `rapidfuzz` (Python) / `fuse.js` (JS) |
| LLM fallback | Claude (`claude-sonnet-4-6`) via Anthropic API |
| Language detection | Unicode range check |
| Storage | Flat JSON files (no DB needed at this scale) |
