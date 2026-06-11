# -*- coding: utf-8 -*-
"""
Build a retrieval gold set from the synthetic Radio Barcelona corpus executable

Derives approximately 40 natural-language questions whose expected_record_ids are computed directly from the deterministic metadata so the gold set is always consistent with the indexed corpus. Mixes thematic multi-target questions with title lookups to provide recall and nDCG measurements
"""

import os
import json
from collections import defaultdict

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA = os.path.join(BASE_PATH, "metadata", "synthetic_radio_barcelona_metadata_v2.jsonl")
OUT = os.path.join(BASE_PATH, "scripts", "rag", "eval_questions.json")

MARK_QUESTIONS = {
    "censorship_stamp":     "radio scripts marked with a censorship stamp",
    "crossed_out_text":     "documents with crossed out or struck through text",
    "handwritten_note":     "pages with handwritten editorial notes in the margin",
    "administrative_stamp": "records with an administrative review stamp",
    "archive_stamp":        "pages with a later archival processing stamp",
    "stamp":                "scripts with a red circular official stamp",
}
TOPIC_QUESTIONS = {
    "fútbol":      "scripts reporting football results and match scores",
    "música":      "music programmes with concerts and song selections",
    "guerra":      "war bulletins and parte de guerra from the civil war",
    "autorizado":  "notices authorizing a broadcast for emission",
    "femenino":    "women's programmes with household advice",
    "infantil":    "children's programmes with songs and contests",
}
ENTITY_QUESTIONS = {
    "Barcelona":  "radio documents mentioning Barcelona",
    "República":  "scripts referring to the Republic",
}


def _year(rec):
    """
    Extract the year from a record identifier
    """
    return rec["record_id"].split("_")[1]


def main():
    """
    Generate evaluation questions from synthetic metadata records
    """
    recs = [json.loads(l) for l in open(METADATA, encoding="utf-8")]
    questions = []

    # 1. Title lookups
    by_title = defaultdict(list)
    for r in recs:
        by_title[r["title"]].append(r["record_id"])
    for title, ids in sorted(by_title.items()):
        questions.append({"question": f"find the {title.lower()}", "expected_record_ids": sorted(ids)})  # Multi-target question generation

    # 2. Mark-type questions
    for mark, q in MARK_QUESTIONS.items():
        ids = sorted(r["record_id"] for r in recs if any(m["mark_type"] == mark for m in r["visual_marks"]))
        if ids:
            questions.append({"question": q, "expected_record_ids": ids})  # Visual mark based question

    # 3. Per-year questions
    years = sorted({_year(r) for r in recs})
    for y in years[::2]:
        ids = sorted(r["record_id"] for r in recs if _year(r) == y)
        questions.append({"question": f"radio scripts from {y}", "expected_record_ids": ids})  # Temporal range question

    # 4. Topic questions
    for kw, q in TOPIC_QUESTIONS.items():
        ids = sorted(r["record_id"] for r in recs
                     if kw.lower() in (r.get("retrieval_text", "") + " " + " ".join(r.get("ocr", {}).get("keywords", []))).lower())
        if ids:
            questions.append({"question": q, "expected_record_ids": ids})  # Content topic question

    # 5. Entity questions
    for ent, q in ENTITY_QUESTIONS.items():
        ids = sorted(r["record_id"] for r in recs if any(e["text"] == ent for e in r.get("entities", [])))
        if ids:
            questions.append({"question": q, "expected_record_ids": ids})  # Entity based question

    # 6. Forensic questions
    for status, q in (("uncertain", "records flagged as uncertain needing human review"),
                      ("verified", "high confidence verified radio scripts")):
        ids = sorted(r["record_id"] for r in recs if r["forensic_flags"]["verification_status"] == status)
        if ids:
            questions.append({"question": q, "expected_record_ids": ids})  # Quality flag question

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(questions)} questions to {OUT}")  # Log output file path
    print(f"Target-set sizes: min={min(len(q['expected_record_ids']) for q in questions)} "
          f"max={max(len(q['expected_record_ids']) for q in questions)}")  # Log target set distribution


if __name__ == "__main__":
    main()
