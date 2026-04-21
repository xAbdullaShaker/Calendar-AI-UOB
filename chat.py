"""
chat.py — UOB Calendar AI CLI chatbot.
FAQ-first matching. RAG retrieval on LLM fallback.

Usage:
    python chat.py
"""

import json
import os
import time
from dotenv import load_dotenv

from core import (
    client,
    SIMILARITY_THRESHOLD, MAX_HISTORY,
    sanitize_input, is_arabic, build_embed_query,
    find_best_faq_match, retrieve_top_chunks,
    build_faq_response, is_date_sensitive,
    ask_llm, load_faq_answers,
)

load_dotenv()


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_embeddings():
    if not os.path.exists("faq_embeddings.json"):
        print("faq_embeddings.json not found. Run embed_faq.py first.")
        exit(1)
    with open("faq_embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["faqs"] if "faqs" in data else data


def load_calendar_chunks():
    if not os.path.exists("calendar_embeddings.json"):
        print("calendar_embeddings.json not found. Run embed_calendar.py first.")
        exit(1)
    with open("calendar_embeddings.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ── Rate limiter (single session for CLI) ─────────────────────────────────────

class RateLimiter:
    def __init__(self, max_calls=30, window_seconds=600):
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps = []

    def is_allowed(self):
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < self.window]
        if len(self._timestamps) >= self.max_calls:
            return False
        self._timestamps.append(now)
        return True

    def seconds_until_reset(self):
        if not self._timestamps:
            return 0
        return max(0, self.window - (time.monotonic() - self._timestamps[0]))


# ── Answer ────────────────────────────────────────────────────────────────────

def answer(question, faq_embeddings, faq_answers, calendar_chunks, history):
    arabic = is_arabic(question)
    embed_query = build_embed_query(question, history)

    try:
        response = client.embeddings.create(
            input=[embed_query],
            model="text-embedding-3-small",
        )
        question_embedding = response.data[0].embedding
    except Exception:
        err = "عذراً، حدث خطأ في الاتصال. حاول مرة أخرى." if arabic else "Sorry, the AI service is unavailable. Please try again."
        return {"response": err}, "[error]"

    best_entry, score = find_best_faq_match(question_embedding, faq_embeddings)

    if score >= SIMILARITY_THRESHOLD and not is_date_sensitive(question):
        source = f"[FAQ match: {score:.0%}]"
        result = build_faq_response(best_entry, arabic, score, faq_answers)
    else:
        top_chunks = retrieve_top_chunks(question_embedding, calendar_chunks)
        source = (
            f"[RAG date-sensitive: {score:.0%}]"
            if score >= SIMILARITY_THRESHOLD
            else f"[RAG fallback: {score:.0%}]"
        )
        result = ask_llm(question, top_chunks, history, arabic=arabic)

    return result, source


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("Loading FAQ embeddings...")
    faq_embeddings = load_embeddings()
    print(f"  {len(faq_embeddings)} FAQ entries loaded")

    print("Loading calendar chunks...")
    calendar_chunks = load_calendar_chunks()
    print(f"  {len(calendar_chunks)} calendar chunks loaded")

    faq_answers = load_faq_answers()

    print("\nUOB Calendar AI — type your question (or 'quit' to exit, 'clear' to reset memory)\n")

    history = []
    limiter = RateLimiter()

    while True:
        question = input("You: ").strip()
        if not question:
            continue

        question, warning = sanitize_input(question)
        if question is None:
            print(f"Bot: {warning}\n")
            continue
        if warning:
            print(f"     {warning}")

        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() in ("clear", "new"):
            history.clear()
            print("Bot: Conversation history cleared.\n")
            continue

        if not limiter.is_allowed():
            wait = int(limiter.seconds_until_reset()) + 1
            print(f"Bot: Too many messages. Please wait {wait} seconds.\n")
            continue

        result, source = answer(question, faq_embeddings, faq_answers, calendar_chunks, history)
        print(f"Bot: {result['response']}")
        print(f"     {source}\n")

        history.append({"question": question, "answer": result["response"]})
        if len(history) > MAX_HISTORY:
            history.pop(0)


if __name__ == "__main__":
    main()
