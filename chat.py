"""
chat.py — UOB Calendar AI chatbot.
Answers from FAQ when possible. Falls back to Cohere LLM with calendar context.

Usage:
    python chat.py
"""

import json
import os
import numpy as np
from dotenv import load_dotenv
import cohere

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))

SIMILARITY_THRESHOLD = 0.75


def load_embeddings():
    if not os.path.exists("faq_embeddings.json"):
        print("faq_embeddings.json not found. Run embed_faq.py first.")
        exit(1)
    with open("faq_embeddings.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_calendar():
    if os.path.exists("uob_calendar.md"):
        with open("uob_calendar.md", "r", encoding="utf-8") as f:
            return f.read()
    return ""


def is_arabic(text):
    return any("\u0600" <= c <= "\u06FF" for c in text)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_best_match(question_embedding, faq_embeddings):
    best_score = 0
    best_entry = None

    for entry in faq_embeddings:
        for emb in entry["embeddings"]:
            score = cosine_similarity(question_embedding, emb)
            if score > best_score:
                best_score = score
                best_entry = entry

    return best_entry, best_score


def ask_llm(question, calendar_context, arabic):
    lang_note = "Respond in Arabic." if arabic else "Respond in English."
    system = f"""You are a helpful assistant for the University of Business (UOB) academic calendar.
Only answer questions using the calendar data provided below.
If the answer is not in the calendar, say you don't have that information.
{lang_note}
Never make up dates or events.
Dates marked with * depend on moon sighting and may shift by ±1 day.

CALENDAR DATA:
{calendar_context}"""

    response = co.chat(
        model="command-r-plus",
        message=question,
        preamble=system,
    )
    return response.text


def answer(question, faq_embeddings, calendar_context):
    arabic = is_arabic(question)

    # Embed the user's question
    response = co.embed(
        texts=[question],
        model="embed-multilingual-v3.0",
        input_type="search_query",
    )
    question_embedding = response.embeddings[0]

    # Find closest FAQ match
    best_entry, score = find_best_match(question_embedding, faq_embeddings)

    if score >= SIMILARITY_THRESHOLD:
        source = f"[FAQ match: {score:.0%}]"
        reply = best_entry["answer_ar"] if arabic else best_entry["answer_en"]
    else:
        source = f"[LLM fallback: best FAQ match was {score:.0%}]"
        reply = ask_llm(question, calendar_context, arabic)

    return reply, source


def main():
    print("Loading embeddings...")
    faq_embeddings = load_embeddings()
    calendar_context = load_calendar()
    print(f"Ready. {len(faq_embeddings)} FAQ entries loaded.\n")
    print("UOB Calendar AI — type your question (or 'quit' to exit)\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        reply, source = answer(question, faq_embeddings, calendar_context)
        print(f"Bot: {reply}")
        print(f"     {source}\n")


if __name__ == "__main__":
    main()
