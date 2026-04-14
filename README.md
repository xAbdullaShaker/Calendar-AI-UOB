# Calendar AI — UOB

A hybrid FAQ chatbot for the UOB academic calendar. Answers in Arabic and English. Uses embeddings for smart matching — the LLM is only a fallback, not the default.

---

## How It Works

```
User asks a question
        ↓
Convert question to embedding (vector)
        ↓
Compare with FAQ question embeddings
        ↓
   ┌────┴────────┐
Match > 75%?   No match?
   │               │
   ✅              ❌
   │               │
Return          Call LLM
pre-written     with calendar
answer          as context
(NO LLM)
```

**~95% of questions never reach the LLM.** The FAQ + embeddings handle them instantly for free.

---

## Why Not Pure RAG?

| | Pure RAG | This Project |
|---|---|---|
| Every question uses LLM | ✅ Yes | ❌ No — FAQ first |
| LLM only for edge cases | ❌ No | ✅ Yes |
| Same question = same answer | ❌ LLM may rephrase | ✅ Always identical |
| Hallucination risk on dates | ⚠️ Yes | ✅ No (pre-written) |
| Cost per common question | 💸 ~$0.001 | 💰 $0.00 |

A calendar has **fixed, factual answers**. Pre-writing them is safer and cheaper than generating them every time.

---

## The Two Things You Prepare

**1. `uob_faq.json` — pre-written answers**
Each entry has multiple phrasings (Arabic + English) and a fixed answer:
```json
{
  "id": "fall_start",
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

**2. `uob_calendar.md` — full calendar data**
Fed to the LLM only when no FAQ match is found.

---

## What Embeddings Do

An embedding converts a sentence into numbers that represent its **meaning** — not its words.

```
"When do classes start?"       → [0.023, -0.451, 0.782, ...]
"متى تبدأ الدراسة؟"            → [0.021, -0.449, 0.779, ...]  ← nearly identical
"When is my first day at uni?" → [0.025, -0.447, 0.780, ...]  ← also close
```

This means Arabic and English questions match each other automatically.

**Setup (once):** Convert all FAQ questions to vectors → save to `faq_embeddings.json`
**Runtime:** Convert user question → compare → return closest answer if similarity > 75%

---

## Real Examples

| User asks | Similarity | LLM used? |
|-----------|-----------|-----------|
| "When does the first semester start?" | 94% | No |
| "متى أول يوم دراسة؟" | 87% | No |
| "If I defer, can I still transfer programs in June?" | 42% | Yes |

---

## Embedding Tools

| Tool | Cost | Arabic |
|------|------|--------|
| OpenAI `text-embedding-3-small` | ~$0.02 / 1M tokens | Excellent |
| Cohere `embed-multilingual` | Free tier | Excellent |
| Sentence Transformers (local) | Free | Good |

Converting all 26 FAQ entries costs **less than $0.001 total**.

---

## Files

| File | Purpose |
|------|---------|
| `uob_faq.json` | FAQ with Arabic + English phrasings and pre-written answers |
| `uob_calendar.md` | Full academic calendar in Markdown — fed to LLM as context |
| `uob_calendar.json` | Same calendar in structured JSON |
| `uob_ai_system_prompt.md` | System prompt restricting AI to calendar data only |
| `docs/architecture.md` | Full technical architecture detail |

---

## Tech Stack

```
Embeddings:  OpenAI text-embedding-3-small  or  Cohere embed-multilingual
Matching:    Cosine similarity on flat JSON (no DB needed at this scale)
LLM:         Claude (claude-sonnet-4-6) with uob_calendar.md as context
Language:    Detected by Arabic Unicode range check
```
