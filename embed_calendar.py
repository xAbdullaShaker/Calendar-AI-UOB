"""
embed_calendar.py — Chunk uob_calendar.md into individual events and embed each one.
Run this once (or whenever the calendar is updated).

Output: calendar_embeddings.json
  [{"chunk": "First Semester (Fall 2025): Mon, 1 Sep 2025 — Faculty members report to duty", "embedding": [...]}, ...]

Usage:
    python embed_calendar.py
"""

import json
import os
import re
from dotenv import load_dotenv
import cohere

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))


def parse_chunks(filepath):
    """Parse uob_calendar.md into one chunk per calendar event row."""
    chunks = []
    current_section = ""

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Detect section headers (## ...)
        if line.startswith("## "):
            current_section = line.lstrip("#").strip()
            continue

        # Detect table rows (| date | event |), skip header and divider rows
        if line.startswith("|") and "---" not in line and "Date" not in line and "Event" not in line:
            # Strip markdown bold and leading/trailing pipes
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            parts = [p.strip() for p in clean.strip("|").split("|")]
            if len(parts) >= 2:
                date = parts[0].strip()
                event = parts[1].strip()
                if date and event:
                    chunk = f"{current_section}: {date} — {event}"
                    chunks.append(chunk)

        # Detect abbreviation lines (Key Abbreviations section)
        if line.startswith("- **") and "—" in line:
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line).lstrip("- ").strip()
            chunk = f"Key Abbreviations: {clean}"
            chunks.append(chunk)

    return chunks


def main():
    chunks = parse_chunks("uob_calendar.md")
    print(f"Parsed {len(chunks)} chunks from uob_calendar.md")

    print("Embedding chunks...")
    response = co.embed(
        texts=chunks,
        model="embed-multilingual-v3.0",
        input_type="search_document",
    )

    results = [
        {"chunk": chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, response.embeddings)
    ]

    with open("calendar_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(results)} chunks to calendar_embeddings.json")
    print("\nSample chunks:")
    for c in results[:3]:
        print(f"  - {c['chunk']}")


if __name__ == "__main__":
    main()
