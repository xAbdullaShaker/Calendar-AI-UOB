"""
embed_faq.py — Run this ONCE to convert FAQ questions to embeddings.
Re-run whenever you update uob_faq.json.

Usage:
    pip install cohere python-dotenv
    python embed_faq.py
"""

import json
import os
from dotenv import load_dotenv
import cohere

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))

with open("uob_faq.json", "r", encoding="utf-8") as f:
    data = json.load(f)

faq = data["faqs"] if "faqs" in data else data
print(f"Embedding {len(faq)} FAQ entries...")

results = []
for entry in faq:
    response = co.embed(
        texts=entry["questions"],
        model="embed-multilingual-v3.0",
        input_type="search_document",
    )
    results.append({
        "id": entry["id"],
        "tags": entry["tags"],
        "questions": entry["questions"],
        "embeddings": response.embeddings,
        "answer_en": entry["answer_en"],
        "answer_ar": entry["answer_ar"],
    })
    print(f"  OK {entry['id']}")

with open("faq_embeddings.json", "w", encoding="utf-8") as f:
    json.dump({"faqs": results}, f, ensure_ascii=False, indent=2)

print(f"\nDone. Saved to faq_embeddings.json")
