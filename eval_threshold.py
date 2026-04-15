"""
eval_threshold.py — Find the optimal SIMILARITY_THRESHOLD for FAQ matching.

Usage:
    python eval_threshold.py
"""

import json
import os
import numpy as np
from dotenv import load_dotenv
import cohere

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ── Reuse helpers from chat.py ────────────────────────────────────────────────

def load_embeddings():
    if not os.path.exists("faq_embeddings.json"):
        print("faq_embeddings.json not found. Run embed_faq.py first.")
        exit(1)
    with open("faq_embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["faqs"] if "faqs" in data else data


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_best_faq_match(question_embedding, faq_embeddings):
    best_score = 0
    best_entry = None
    for entry in faq_embeddings:
        for emb in entry["embeddings"]:
            score = cosine_similarity(question_embedding, emb)
            if score > best_score:
                best_score = score
                best_entry = entry
    return best_entry, best_score


# ── Test cases ────────────────────────────────────────────────────────────────
# "expected": FAQ id string  →  should hit FAQ
# "expected": None           →  should fall through to LLM

TEST_CASES = [
    # --- Exact / near-exact matches ---
    {"question": "hi",                                         "expected": "greeting"},
    {"question": "hello",                                      "expected": "greeting"},
    {"question": "when does the first semester start",         "expected": "fall_start"},
    {"question": "when does the second semester start",        "expected": "spring_start"},
    {"question": "when does summer session begin",             "expected": "summer_start"},
    {"question": "when does first semester end",               "expected": "fall_end"},
    {"question": "when are fall finals",                       "expected": "fall_finals"},
    {"question": "when is Eid Al Fitr",                        "expected": "eid_fitr"},
    {"question": "when is national day",                       "expected": "national_day"},
    {"question": "last day to drop with refund",               "expected": "last_day_drop_refund"},

    # --- Paraphrases (different wording, same intent) ---
    {"question": "what day do classes kick off this fall",     "expected": "fall_start"},
    {"question": "what is the deadline to withdraw from a course with W grade", "expected": "fall_withdrawal_w"},
    {"question": "how long is the add/drop period in spring",  "expected": "spring_drop_add"},
    {"question": "when do semester grades come out",           "expected": "fall_results"},
    {"question": "what holidays does UOB have this year",      "expected": "all_holidays"},
    {"question": "when can I appeal my grade",                 "expected": "grade_appeal"},
    {"question": "when is the transfer application period",    "expected": "transfer_period"},
    {"question": "when are admission tests scheduled",         "expected": "admission_tests"},
    {"question": "what is the tuition fee",                    "expected": "tuition_fees"},
    {"question": "when does preliminary registration open",    "expected": "preliminary_registration"},

    # --- Arabic questions ---
    {"question": "متى يبدأ الفصل الأول",                      "expected": "fall_start"},
    {"question": "مرحبا",                                      "expected": "greeting"},
    {"question": "متى يبدأ الفصل الصيفي",                     "expected": "summer_start"},
    {"question": "متى ينتهي الفصل الأول",                     "expected": "fall_end"},
    {"question": "متى تبدأ الدراسة للفصل الثاني",             "expected": "spring_start"},
    {"question": "متى الإجازات الرسمية",                       "expected": "all_holidays"},

    # --- Mixed Arabic/English ---
    {"question": "متى fall semester يبدأ",                    "expected": "fall_start"},
    {"question": "when is عيد الفطر",                         "expected": "eid_fitr"},

    # --- Out-of-scope (should fall through to LLM) ---
    {"question": "how do I reset my UOB portal password",      "expected": None},
    {"question": "where is the library on campus",             "expected": None},
    {"question": "who is the president of UOB",                "expected": None},
    {"question": "what engineering majors does UOB offer",     "expected": None},
    {"question": "how many credits do I need to graduate",     "expected": None},

    # --- Edge cases ---
    {"question": "?",                                          "expected": None},
    {"question": "uob",                                        "expected": None},
    {"question": "tomorrow",                                   "expected": None},
]


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(faq_embeddings, embeddings_cache, thresholds):
    results = {}

    for threshold in thresholds:
        correct = 0
        errors = []

        for tc in TEST_CASES:
            q = tc["question"]
            expected = tc["expected"]
            q_emb = embeddings_cache[q]

            best_entry, score = find_best_faq_match(q_emb, faq_embeddings)
            matched_id = best_entry["id"] if score >= threshold else None

            if matched_id == expected:
                correct += 1
            else:
                errors.append({
                    "question": q,
                    "expected": expected,
                    "got": matched_id,
                    "score": round(score, 4),
                })

        accuracy = correct / len(TEST_CASES)
        results[threshold] = {"accuracy": accuracy, "errors": errors}

    return results


def main():
    print("Loading FAQ embeddings...")
    faq_embeddings = load_embeddings()
    print(f"  {len(faq_embeddings)} FAQ entries loaded")

    questions = [tc["question"] for tc in TEST_CASES]
    print(f"\nEmbedding {len(questions)} test questions (single API call)...")
    response = co.embed(
        texts=questions,
        model="embed-multilingual-v3.0",
        input_type="search_query",
    )
    embeddings_cache = {q: emb for q, emb in zip(questions, response.embeddings)}
    print("  Done.\n")

    thresholds = [round(t, 2) for t in np.arange(0.50, 0.86, 0.05)]
    results = evaluate(faq_embeddings, embeddings_cache, thresholds)

    # ── Print results table ───────────────────────────────────────────────────
    total = len(TEST_CASES)
    print(f"{'Threshold':<12} {'Correct':<10} {'Accuracy':<10} {'Errors'}")
    print("-" * 55)

    best_threshold = None
    best_accuracy = -1

    for t in thresholds:
        r = results[t]
        acc = r["accuracy"]
        n_correct = round(acc * total)
        n_errors = total - n_correct
        marker = " ◄ best" if acc > best_accuracy else ""

        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t

        print(f"  {t:<10.2f} {n_correct:<10} {acc*100:<9.1f}%  {n_errors}{marker}")

    # ── Show errors at best threshold ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Recommended threshold: {best_threshold}  (accuracy {best_accuracy*100:.1f}%)")
    print(f"{'='*55}")

    errors = results[best_threshold]["errors"]
    if errors:
        print(f"\nMisclassified at {best_threshold} ({len(errors)} cases):\n")
        for e in errors:
            exp = e["expected"] or "LLM"
            got = e["got"] or "LLM"
            print(f"  Q : {e['question']}")
            print(f"      expected={exp:<25} got={got:<25} score={e['score']}")
    else:
        print("\nPerfect score — no misclassifications at this threshold.")


if __name__ == "__main__":
    main()
