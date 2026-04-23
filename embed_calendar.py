"""
embed_calendar.py — Chunk uob_calendar.md into individual events and embed each one.

This script reads the academic calendar markdown file, splits it into one chunk
per event row, appends Arabic keyword annotations so Arabic queries can find
English calendar text, then sends everything to OpenAI for embedding.

The output (calendar_embeddings.json) is what the server searches when the LLM
path is triggered — the top 4 most relevant chunks are passed to the LLM as context.

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
from openai import OpenAI

# Load OPENAI_API_KEY from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Arabic keyword annotations ────────────────────────────────────────────────
#
# The calendar is written entirely in English. If a student asks in Arabic,
# their query vector won't match the English calendar chunks well.
#
# Solution: append Arabic keyword phrases to each chunk before embedding,
# so Arabic queries land close to the right calendar events in vector space.
#
# Format: ([english_keywords_to_match], "arabic_annotation_to_append")
#
ARABIC_ANNOTATIONS = [
    # Withdrawal
    (["withdrawal period", "withdrawal with", "w grade", "w\" grade"],
     "فترة الانسحاب / انسحاب بدرجة W / آخر موعد الانسحاب / الانسحاب من المقررات"),
    (["mandatory withdrawal", "wa/wf"],
     "الانسحاب الإجباري / WA/WF / آخر موعد رصد الانسحاب الإجباري"),
    # Registration & Add/Drop
    (["registration, drop & add", "drop & add", "drop and add"],
     "التسجيل والحذف والإضافة / الاد والدروب / فترة التسجيل"),
    (["preliminary registration"],
     "التسجيل المبدئي / التسجيل المبدئي للفصل"),
    # Classes
    (["classes begin"],
     "بدء الدراسة / أول يوم دراسي / بداية الفصل الدراسي"),
    (["last day of classes"],
     "آخر يوم دراسي / آخر يوم في الدراسة / نهاية الدراسة"),
    # Exams
    (["final exams period", "final exams"],
     "الامتحانات النهائية / فترة الامتحانات / الفاينل / موعد الامتحانات"),
    (["last day to submit final", "submit final exam grades", "submit final grades"],
     "آخر موعد تسليم الدرجات / تسليم درجات الامتحانات"),
    # Results
    (["final results announced", "results announced"],
     "إعلان النتائج النهائية / نتائج الفصل / إعلان النتائج"),
    # Drop with refund
    (["last day to drop courses with fee refund", "fee refund"],
     "آخر موعد حذف المقررات واسترجاع الرسوم / استرجاع الرسوم الدراسية"),
    # Deferral
    (["semester deferral", "deferral"],
     "تأجيل الدراسة / طلب تأجيل فصل / تأجيل فصل أو فصلين"),
    # Holidays
    (["national day", "accession day"],
     "العيد الوطني / عيد الجلوس / إجازة العيد الوطني"),
    (["eid al-fitr"],
     "عيد الفطر / إجازة عيد الفطر المبارك"),
    (["eid al-adha"],
     "عيد الأضحى / إجازة عيد الأضحى المبارك"),
    (["ramadan"],
     "رمضان / شهر رمضان / توقيت رمضان"),
    (["prophet muhammad", "prophet's birthday"],
     "المولد النبوي / إجازة المولد النبوي الشريف"),
    (["ashura"],
     "عاشوراء / إجازة عاشوراء"),
    (["islamic new year"],
     "رأس السنة الهجرية / إجازة رأس السنة الهجرية"),
    (["new year's day", "new year"],
     "رأس السنة الميلادية / إجازة رأس السنة"),
    (["labour day"],
     "يوم العمال / إجازة يوم العمال"),
    # Grade appeals
    (["grade-appeal", "grade appeal", "appeal submissions"],
     "التظلم من نتيجة مقرر / طلبات التظلم / فترة التظلم"),
    # Faculty
    (["faculty members report", "faculty report"],
     "دوام أعضاء هيئة التدريس / بدء دوام الأساتذة"),
    (["faculty break", "faculty duty"],
     "إجازة ما بين الفصلين / آخر يوم دوام هيئة التدريس"),
    # Advising
    (["academic advising"],
     "الإرشاد الأكاديمي / الإرشاد للتسجيل المبدئي"),
    # Minor specialization
    (["minor-specialization"],
     "التخصصات الفرعية / طلبات التخصص الفرعي"),
    # Transfer
    (["transfer requests"],
     "طلبات التحويل / فترة التحويل بين البرامج"),
    # Admission
    (["admission tests", "interviews"],
     "اختبارات القبول / مقابلات القبول"),
    # English test
    (["english test", "english language test", "ielts", "toefl"],
     "امتحان اللغة الإنجليزية / اختبار الإنجليزية للدراسات العليا"),
    # Tuition
    (["tuition fees", "pay tuition"],
     "الرسوم الدراسية / آخر موعد دفع الرسوم"),
    # Course evaluation
    (["course evaluation"],
     "تقييم المقررات الدراسية / تقييم المقررات من قبل الطلبة"),
    # Case studies
    (["case study", "case-study"],
     "دراسة حالة طلابية / طلبات دراسة الحالة"),
    # Sections
    (["first semester", "fall 2025"],
     "الفصل الدراسي الأول / الفصل الأول"),
    (["second semester", "spring 2026"],
     "الفصل الدراسي الثاني / الفصل الثاني / الربيع"),
    (["summer session"],
     "الفصل الدراسي الصيفي / الفصل الصيفي"),
]


def annotate_arabic(event_text):
    """
    Check if an event text matches any annotation rules and return the Arabic keywords.

    Looks at the English event text and finds all matching annotation entries.
    Returns the Arabic phrases joined by ' | ', or an empty string if no match.
    These Arabic phrases are appended to the chunk before embedding.
    """
    event_lower = event_text.lower()
    annotations = []
    for keywords, arabic in ARABIC_ANNOTATIONS:
        # If any of the English keywords appear in this event, add the Arabic annotation
        if any(kw in event_lower for kw in keywords):
            annotations.append(arabic)
    return " | ".join(annotations) if annotations else ""


def parse_chunks(filepath):
    """
    Parse uob_calendar.md and return a list of bilingual text chunks.

    Each chunk represents one calendar event row. The format is:
        "Section Name: Date — Event Description [Arabic keywords]"

    The section name is taken from the nearest ## heading above the row.
    Arabic keywords are appended where the annotation rules match.
    """
    chunks = []
    current_section = ""  # tracks which semester/section we're currently inside

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # ── Section headers (## First Semester, ## Second Semester, etc.) ──────
        if line.startswith("## "):
            # Update the current section name (strip the ## prefix)
            current_section = line.lstrip("#").strip()
            continue

        # ── Table rows (| date | event |) ────────────────────────────────────
        # Skip header rows ("Date", "Event") and divider rows ("---")
        if line.startswith("|") and "---" not in line and "Date" not in line and "Event" not in line:
            # Remove markdown bold markers (**text** → text)
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            # Split the row by | and strip whitespace from each cell
            parts = [p.strip() for p in clean.strip("|").split("|")]
            if len(parts) >= 2:
                date = parts[0].strip()
                event = parts[1].strip()
                if date and event:
                    # Look up Arabic keywords to append for cross-language matching
                    arabic = annotate_arabic(event)
                    # Build the final chunk text: "Section: Date — Event [Arabic]"
                    chunk = f"{current_section}: {date} — {event}"
                    if arabic:
                        chunk += f" [{arabic}]"
                    chunks.append(chunk)

        # ── Abbreviation definitions (Key Abbreviations section) ─────────────
        # Lines like: - **W** — Student-initiated withdrawal
        if line.startswith("- **") and "—" in line:
            # Remove bold markers and the leading "- "
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line).lstrip("- ").strip()
            chunk = f"Key Abbreviations: {clean}"
            chunks.append(chunk)

    return chunks


def main():
    """
    Main function — parse the calendar, embed all chunks, and save to JSON.
    """
    # Parse the calendar markdown into individual event chunks
    chunks = parse_chunks("uob_calendar.md")
    print(f"Parsed {len(chunks)} chunks from uob_calendar.md")

    # Send all chunks to OpenAI in one API call — more efficient than one-by-one
    print("Embedding chunks...")
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-large",  # 3072-dimension model — better Arabic understanding
    )

    # Pair each chunk text with its embedding vector
    results = [
        {"chunk": chunk, "embedding": r.embedding}
        for chunk, r in zip(chunks, response.data)
    ]

    # Save to disk — this file is loaded at server startup for numpy mode,
    # or uploaded to Supabase via migrate_to_pgvector.py for production mode
    with open("calendar_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(results)} chunks to calendar_embeddings.json")

    # Show a few examples so you can verify the output looks correct
    print("\nSample chunks:")
    for c in results[:3]:
        print(f"  - {c['chunk']}")


if __name__ == "__main__":
    main()
