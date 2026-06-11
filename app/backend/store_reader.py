# -*- coding: utf-8 -*-
"""
Read access to the indexed stores for the viewer (components)

Reads the record JSON files written by the indexing stage and maps them into the
compact viewer models, and resolves the per-page derived image URLs

This is filesystem-backed (outputs/index/records + outputs/derived); a MongoDB-backed
reader can later implement the same two functions without touching the API layer
"""

import os
import json
import glob
from typing import List, Optional

from app.backend.schemas import DocumentSummary, PageDetail, RegionView, MarkView

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIR_RECORDS = os.path.join(PATH_ROOT, "outputs", "index", "records")
PATH_DIR_DERIVED = os.path.join(PATH_ROOT, "outputs", "derived")

# Derived image suffix -> viewer layer name
LAYER_FILES = {
    "printed":     "_pr.png",
    "handwritten": "_hw.png",
    "noise":       "_ns.png",
    "heatmap":     "_conf.png",
    "overlay":     "_overlay.png",
}


def _doc_dir(doc_id):
    return os.path.join(PATH_DIR_RECORDS, doc_id)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_documents() -> List[DocumentSummary]:
    """Lists every indexed document with its page ids"""
    out = []
    if not os.path.isdir(PATH_DIR_RECORDS):
        return out
    for doc_id in sorted(os.listdir(PATH_DIR_RECORDS)):
        d = _doc_dir(doc_id)
        if not os.path.isdir(d):
            continue
        pages = sorted({os.path.basename(p)[:-len(".page.json")]
                        for p in glob.glob(os.path.join(d, "*.page.json"))})
        out.append(DocumentSummary(document_id=doc_id, n_pages=len(pages), pages=pages))
    return out


def get_page(doc_id, page_id) -> Optional[PageDetail]:
    """Assembles the viewer detail for one page, or None if the page record is missing"""
    d = _doc_dir(doc_id)
    page_path = os.path.join(d, f"{page_id}.page.json")
    if not os.path.exists(page_path):
        return None
    page = _read_json(page_path)

    printed_regions, handwritten_regions = [], []
    for rp in sorted(glob.glob(os.path.join(d, f"{page_id}.*.json"))):
        if rp.endswith(".page.json"):
            continue
        rec = _read_json(rp)
        rid = rec.get("region_id") or ""
        layer = "printed" if rid.startswith("pr") else "handwritten" if rid.startswith("hw") else "other"
        flags = rec.get("forensic_flags", {})
        ocr = rec.get("ocr", {})
        rv = RegionView(
            region_id=rid, layer=layer,
            text=ocr.get("full_text", ""),
            confidence=flags.get("confidence_score", 0.0),
            verification_status=flags.get("verification_status", "uncertain"),
            bbox=rec.get("bbox", []),
            reading_order=rec.get("reading_order", 0),
            word_confidences=ocr.get("word_confidences", []),
        )
        (printed_regions if layer == "printed" else handwritten_regions).append(rv)

    # Present regions in column-aware reading order (set at extraction time)
    printed_regions.sort(key=lambda r: r.reading_order)
    handwritten_regions.sort(key=lambda r: r.reading_order)

    marks = [MarkView(mark_id=m.get("mark_id", ""), mark_type=m.get("mark_type", ""),
                      bbox=m.get("bbox", []), confidence=m.get("confidence_score", 0.0))
             for m in page.get("visual_marks", [])]

    derived_dir = os.path.join(PATH_DIR_DERIVED, doc_id)
    layers = {}
    for name, suffix in LAYER_FILES.items():
        if os.path.exists(os.path.join(derived_dir, f"{page_id}{suffix}")):
            layers[name] = f"/api/derived/{doc_id}/{page_id}{suffix}"

    forensic = page.get("forensic_flags", {})
    return PageDetail(
        document_id=doc_id,
        page_id=page_id,
        title=page.get("title", ""),
        date_created=page.get("date_created"),
        language=page.get("language", []),
        printed_text=" ".join(r.text for r in printed_regions if r.text),
        handwritten_text=" ".join(r.text for r in handwritten_regions if r.text),
        printed_regions=printed_regions,
        handwritten_regions=handwritten_regions,
        marks=marks,
        forensic=forensic,
        confidence=forensic.get("confidence_score", 0.0),
        layers=layers,
    )
