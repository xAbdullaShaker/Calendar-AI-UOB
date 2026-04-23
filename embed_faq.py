"""
embed_faq.py — Run this ONCE to convert FAQ questions to embeddings.

This script reads every question from uob_faq.json, sends them to OpenAI's
embedding API, and saves the resulting vectors to faq_embeddings.json.

Those vectors are what the live server uses to match user questions to FAQ answers.
Re-run this script any time you add or change questions in uob_faq.json.

Usage:
    pip install openai python-dotenv
    python embed_faq.py
"""

import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load the OPENAI_API_KEY from the .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def normalize_arabic(text):
    """
    Apply basic Arabic character normalization before embedding.

    This makes sure spelling variations of the same word map to the same vector.
    For example 'أول' and 'اول' both become 'اول' after normalization,
    so they embed close to each other.
    """
    # Unify all Alef variants (أ, إ, آ) to plain Alef (ا)
    text = re.sub(r'[أإآ]', 'ا', text)
    # Unify Taa marbuta (ة) to Haa (ه) — common end-of-word variation
    text = re.sub(r'ة', 'ه', text)
    # Unify Alef maqsura (ى) to Alef (ا)
    text = re.sub(r'ى', 'ا', text)
    # Remove Arabic diacritics (tashkeel) — short vowel marks that vary between writers
    text = re.sub(r'[\u064B-\u065F]', '', text)
    return text


# ── Load the FAQ data ─────────────────────────────────────────────────────────

with open("uob_faq.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Handle both formats: {"faqs": [...]} and [...]
faq = data["faqs"] if "faqs" in data else data
print(f"Embedding {len(faq)} FAQ entries...")

# ── Generate embeddings for each FAQ entry ────────────────────────────────────

results = []
for entry in faq:
    # Normalize each question variant before sending to OpenAI
    # This ensures the stored vectors match what normalize_arabic() produces at query time
    normalized_questions = [normalize_arabic(q) for q in entry["questions"]]

    # Send all question variants for this FAQ entry to OpenAI in one API call
    response = client.embeddings.create(
        input=normalized_questions,
        model="text-embedding-3-large",  # 3072-dimension model — better Arabic understanding
    )

    # Extract just the vector (list of floats) from each result
    embeddings = [r.embedding for r in response.data]

    # Store the full entry along with its embeddings
    results.append({
        "id": entry["id"],                  # unique identifier, e.g. "fall_start"
        "tags": entry["tags"],              # topic tags, e.g. ["semester-start", "fall"]
        "questions": entry["questions"],    # original question variants (not normalized)
        "embeddings": embeddings,           # one vector per question variant
        "answer_en": entry["answer_en"],    # pre-written English answer
        "answer_ar": entry["answer_ar"],    # pre-written Arabic answer
    })
    print(f"  OK {entry['id']}")

# ── Save results to disk ──────────────────────────────────────────────────────

# Write all embeddings to faq_embeddings.json — this is what the server loads at startup
with open("faq_embeddings.json", "w", encoding="utf-8") as f:
    json.dump({"faqs": results}, f, ensure_ascii=False, indent=2)

print(f"\nDone. Saved to faq_embeddings.json")
