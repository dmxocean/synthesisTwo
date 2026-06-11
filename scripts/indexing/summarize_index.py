# -*- coding: utf-8 -*-
"""
Aggregation engine for indexed record summaries

This module reads page and region records from the indexing stage and generates a consolidated metrics JSON. It calculates confidence distributions, verification statuses, and visual mark frequencies across documents and historical epochs
"""

import os
import re
import glob
import json
import argparse
from collections import Counter, defaultdict

import numpy as np

from src.core.config import PATH_DIR_RECORDS, get_artifact_dir, EPOCHS, DICT_EPOCHS

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def epoch_of_doc(doc_id):
    """
    Determine the historical epoch based on the document identifier year
    """
    m = re.search(r"a(\d{4})", doc_id)
    if m:
        y = int(m.group(1))
        for name, rng in DICT_EPOCHS.items():
            if y in rng:
                return name
    return "unknown"


def _hist(values, bins=10, lo=0.0, hi=1.0):
    """
    Calculate histogram bins and counts for a distribution
    """
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi)) if values else (np.zeros(bins), np.linspace(lo, hi, bins + 1))
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    return {"bin_center": [float(c) for c in centers], "count": [int(c) for c in counts]}


def main(args):
    """
    Aggregate indexing metrics across all page and region records

    Args:
        args (argparse.Namespace): Command line arguments containing the records root path
    """
    page_files = sorted(glob.glob(os.path.join(args.records, "*", "*.page.json")))
    region_files = [p for p in glob.glob(os.path.join(args.records, "*", "*.json")) if not p.endswith(".page.json")]
    if not page_files:
        print(f"[!] No page records under {args.records} (run build_index first)", flush=True)

    confs, marks, alerts = [], Counter(), Counter()
    verif = Counter()
    n_review = 0
    regions_per_page = []
    by_epoch = defaultdict(lambda: {"n_pages": 0, "conf": [], "marks": Counter(), "alerts": Counter()})
    by_doc = defaultdict(lambda: {"n_pages": 0, "n_regions": 0, "n_marks": 0, "conf": []})

    regions_by_page = Counter()
    for rp in region_files:
        rec = json.load(open(rp, encoding="utf-8"))
        regions_by_page[rec.get("page_id", "")] += 1  # Track region count per page
        by_doc[rec.get("document_id", "")]["n_regions"] += 1

    for pf in page_files:
        rec = json.load(open(pf, encoding="utf-8"))
        doc = rec.get("document_id", "")
        ep = epoch_of_doc(doc)
        ff = rec.get("forensic_flags", {})
        c = float(ff.get("confidence_score", 0.0))
        confs.append(c)
        verif[ff.get("verification_status", "uncertain")] += 1
        n_review += 1 if ff.get("human_review_required") else 0
        for a in ff.get("alerts", []):
            alerts[a] += 1
            by_epoch[ep]["alerts"][a] += 1
        for m in rec.get("visual_marks", []):
            marks[m.get("mark_type", "unknown")] += 1
            by_epoch[ep]["marks"][m.get("mark_type", "unknown")] += 1
        regions_per_page.append(regions_by_page.get(rec.get("page_id", ""), 0))
        by_epoch[ep]["n_pages"] += 1
        by_epoch[ep]["conf"].append(c)
        by_doc[doc]["n_pages"] += 1
        by_doc[doc]["n_marks"] += len(rec.get("visual_marks", []))
        by_doc[doc]["conf"].append(c)

    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    out = {
        "stage": "index_summary",
        "n_documents": len(by_doc),
        "n_pages": len(page_files),
        "n_regions": len(region_files),
        "n_records": len(page_files) + len(region_files),
        "overall": {
            "verification": dict(verif),
            "human_review_required": n_review,
            "confidence_mean": _mean(confs),
            "confidence_median": float(np.median(confs)) if confs else 0.0,
            "confidence_hist": _hist(confs),
            "mark_types": dict(marks),
            "alerts": dict(alerts),
            "regions_per_page_hist": {str(k): int(v) for k, v in sorted(Counter(regions_per_page).items())},
        },
        "per_epoch": {ep: {"n_pages": d["n_pages"], "confidence_mean": _mean(d["conf"]),
                           "mark_types": dict(d["marks"]), "alerts": dict(d["alerts"])}
                      for ep, d in by_epoch.items()},
        "per_document": {doc: {"n_pages": d["n_pages"], "n_regions": d["n_regions"],
                               "n_marks": d["n_marks"], "confidence_mean": _mean(d["conf"])}
                         for doc, d in sorted(by_doc.items())},
    }
    out_path = os.path.join(get_artifact_dir("indexing", "records", "metrics"), "index_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[*] Wrote {out_path}", flush=True)
    print(f"[*] docs={out['n_documents']} pages={out['n_pages']} regions={out['n_regions']} "
          f"mean_conf={out['overall']['confidence_mean']:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Summarize the indexed records into outputs/indexing/records/metrics/")
    p.add_argument("--records", default=PATH_DIR_RECORDS, help="records root (outputs/index/records)")
    main(p.parse_args())
