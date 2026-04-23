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
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Arabic keyword annotations for common calendar events.
# These are appended to each chunk so Arabic queries embed close enough
# to match the English calendar text during cosine similarity search.
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
    """Return Arabic keywords to append to a chunk based on its English content."""
    event_lower = event_text.lower()
    annotations = []
    for keywords, arabic in ARABIC_ANNOTATIONS:
        if any(kw in event_lower for kw in keywords):
            annotations.append(arabic)
    return " | ".join(annotations) if annotations else ""


def parse_chunks(filepath):
    """Parse uob_calendar.md into one bilingual chunk per calendar event row."""
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
                    arabic = annotate_arabic(event)
                    chunk = f"{current_section}: {date} — {event}"
                    if arabic:
                        chunk += f" [{arabic}]"
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
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-large",
    )

    results = [
        {"chunk": chunk, "embedding": r.embedding}
        for chunk, r in zip(chunks, response.data)
    ]

    with open("calendar_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(results)} chunks to calendar_embeddings.json")
    print("\nSample chunks:")
    for c in results[:3]:
        print(f"  - {c['chunk']}")


if __name__ == "__main__":
    main()
