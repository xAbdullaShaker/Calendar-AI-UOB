"""
chat.py — UOB Calendar AI CLI chatbot.

This is the terminal (command-line) version of the same AI assistant.
It uses the exact same logic as api.py but runs as an interactive chat
in the terminal instead of a web server.

Useful for testing and debugging without running the full web stack.

Usage:
    python chat.py
"""

import json
import os
import time
from dotenv import load_dotenv

# Import shared logic from core.py — same functions used by the web API
from core import (
    client,
    SIMILARITY_THRESHOLD, MAX_HISTORY,
    sanitize_input, is_arabic, build_embed_query,
    find_best_faq_match, retrieve_top_chunks,
    build_faq_response, is_date_sensitive,
    ask_llm, load_faq_answers,
)

# Load environment variables (OPENAI_API_KEY etc.) from .env file
load_dotenv()


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_embeddings():
    """
    Load pre-computed FAQ embeddings from the local JSON file.
    Exits with an error if the file doesn't exist (user needs to run embed_faq.py first).
    """
    if not os.path.exists("faq_embeddings.json"):
        print("faq_embeddings.json not found. Run embed_faq.py first.")
        exit(1)
    with open("faq_embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # Handle both file formats: {"faqs": [...]} and [...]
    return data["faqs"] if "faqs" in data else data


def load_calendar_chunks():
    """
    Load pre-computed calendar chunk embeddings from the local JSON file.
    Exits with an error if the file doesn't exist (user needs to run embed_calendar.py first).
    """
    if not os.path.exists("calendar_embeddings.json"):
        print("calendar_embeddings.json not found. Run embed_calendar.py first.")
        exit(1)
    with open("calendar_embeddings.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ── Rate limiter (single session for CLI) ─────────────────────────────────────

class RateLimiter:
    """
    Prevents too many API calls in a short time period.
    Allows up to max_calls requests within any rolling window_seconds window.
    """
    def __init__(self, max_calls=30, window_seconds=600):
        self.max_calls = max_calls        # max messages allowed
        self.window = window_seconds       # time window in seconds (600 = 10 minutes)
        self._timestamps = []              # list of recent message timestamps

    def is_allowed(self):
        """
        Check if a new message is allowed.
        Removes timestamps older than the window, then checks if we're under the limit.
        Returns True if allowed, False if rate limit exceeded.
        """
        now = time.monotonic()
        # Keep only timestamps within the current window
        self._timestamps = [t for t in self._timestamps if now - t < self.window]
        if len(self._timestamps) >= self.max_calls:
            return False  # too many recent messages
        # Record this message's timestamp
        self._timestamps.append(now)
        return True

    def seconds_until_reset(self):
        """
        Return how many seconds until the oldest message falls outside the window.
        Used to tell the user how long to wait.
        """
        if not self._timestamps:
            return 0
        return max(0, self.window - (time.monotonic() - self._timestamps[0]))


# ── Answer ────────────────────────────────────────────────────────────────────

def answer(question, faq_embeddings, faq_answers, calendar_chunks, history):
    """
    Process a user question and return an answer using the FAQ-first pipeline.

    Steps:
    1. Detect language (Arabic or English)
    2. Build the embed query (with follow-up expansion and normalization)
    3. Embed the question using OpenAI
    4. Find the closest FAQ match
    5. If good match and not date-sensitive → return FAQ answer
    6. Otherwise → retrieve top calendar chunks and ask the LLM

    Returns a tuple of (result_dict, source_label).
    Note: chat.py still uses the old single-match logic (find_best_faq_match).
          The domain guard and ambiguity check are in api.py only.
    """
    # Detect if the question is primarily Arabic or English
    arabic = is_arabic(question)

    # Build the normalized query for embedding (handles follow-ups, dialect, typos)
    embed_query = build_embed_query(question, history)

    # Convert the normalized question to a vector using OpenAI
    try:
        response = client.embeddings.create(
            input=[embed_query],
            model="text-embedding-3-small",  # CLI uses small model for lower cost
        )
        question_embedding = response.data[0].embedding
    except Exception:
        # Return a friendly error message if the API call fails
        err = "عذراً، حدث خطأ في الاتصال. حاول مرة أخرى." if arabic else "Sorry, the AI service is unavailable. Please try again."
        return {"response": err}, "[error]"

    # Find the single best-matching FAQ entry by cosine similarity
    best_entry, score = find_best_faq_match(question_embedding, faq_embeddings)

    if score >= SIMILARITY_THRESHOLD and not is_date_sensitive(question):
        # ── FAQ path: good match found and not time-dependent ─────────────────
        source = f"[FAQ match: {score:.0%}]"
        result = build_faq_response(best_entry, arabic, score, faq_answers)
    else:
        # ── RAG path: no good match, or question needs today's date ──────────
        # Retrieve the top 4 most relevant calendar event chunks
        top_chunks = retrieve_top_chunks(question_embedding, calendar_chunks)
        # Label explains why RAG was used
        source = (
            f"[RAG date-sensitive: {score:.0%}]"
            if score >= SIMILARITY_THRESHOLD
            else f"[RAG fallback: {score:.0%}]"
        )
        # Ask the LLM with those chunks as context
        result = ask_llm(question, top_chunks, history, arabic=arabic)

    return result, source


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    """
    Start the interactive CLI chatbot.
    Loads all data, then runs a loop reading questions from the terminal.
    """
    # Load all pre-computed data into memory
    print("Loading FAQ embeddings...")
    faq_embeddings = load_embeddings()
    print(f"  {len(faq_embeddings)} FAQ entries loaded")

    print("Loading calendar chunks...")
    calendar_chunks = load_calendar_chunks()
    print(f"  {len(calendar_chunks)} calendar chunks loaded")

    faq_answers = load_faq_answers()

    print("\nUOB Calendar AI — type your question (or 'quit' to exit, 'clear' to reset memory)\n")

    history = []              # stores recent turns for follow-up context
    limiter = RateLimiter()   # rate limiter for this CLI session

    # ── Main input loop ───────────────────────────────────────────────────────
    while True:
        question = input("You: ").strip()

        # Skip empty input
        if not question:
            continue

        # Clean and validate the input
        question, warning = sanitize_input(question)
        if question is None:
            # Input was rejected (gibberish, symbols only, etc.)
            print(f"Bot: {warning}\n")
            continue
        if warning:
            # Input was accepted but with a warning (e.g. truncated)
            print(f"     {warning}")

        # Handle quit commands
        if question.lower() in ("quit", "exit", "q"):
            break

        # Handle memory reset command
        if question.lower() in ("clear", "new"):
            history.clear()
            print("Bot: Conversation history cleared.\n")
            continue

        # Check rate limit before calling the API
        if not limiter.is_allowed():
            wait = int(limiter.seconds_until_reset()) + 1
            print(f"Bot: Too many messages. Please wait {wait} seconds.\n")
            continue

        # Get the answer and print it
        result, source = answer(question, faq_embeddings, faq_answers, calendar_chunks, history)
        print(f"Bot: {result['response']}")
        print(f"     {source}\n")

        # Add this turn to conversation history for follow-up detection
        history.append({"question": question, "answer": result["response"]})

        # Keep only the last MAX_HISTORY turns to avoid unbounded memory growth
        if len(history) > MAX_HISTORY:
            history.pop(0)


if __name__ == "__main__":
    main()
