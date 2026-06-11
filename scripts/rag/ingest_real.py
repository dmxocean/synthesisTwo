# -*- coding: utf-8 -*-
"""
Real-records -> RAG corpus bridge (executable workflow)

Consolidates the per-record JSON files written by the indexing run into ONE canonical
JSONL that the RAG stage ingests. This is the real counterpart of generate_metadata.py
(which emits synthetic/fake records) - it never invents data, it only reads what the
pipeline produced and enriches each record with the display-only archival block derived
from its document_id.

Flow:
  outputs/index/records/{doc}/{page}.{unit}.json   (source of truth, also backs the app)
        |  ingest_real.py  (this script; stdlib only, no GPU / no Qdrant deps)
        v
  outputs/rag/metadata/records.jsonl               (one JSON object per line)
        |  rag.py build --input … --collection radio_barcelona_real
        v
  Qdrant collection of embedded records

Every record is ingested faithfully - all three layers carry signal (printed OCR,
handwritten HTR, and noise marks such as stamps / crosses / lines), so the bridge does
not drop records on a text-only heuristic. The title/date_created fields are already set
on disk by the builder; here we add the collection / archival_metadata / physical_metadata
blocks that RadioArchiveMetadataReader reads (with .get()) for provenance display. None of
these are router filter fields.

Run from the project root:
  python scripts/rag/ingest_real.py
"""

import os
import glob
import json
import argparse
from collections import Counter

from src.core.config import PATH_DIR_RECORDS, PATH_FILE_RAG_CORPUS
from src.core.archival import derive_archival, derive_epoch, parse_document_id


def _enrich(record):
    """Merges the display-only archival block into a record dict (in place) and returns it."""
    doc_id = record.get("document_id", "")
    record.update(derive_archival(doc_id))
    src_path = (record.get("provenance") or {}).get("source_image_path", "")
    record["physical_metadata"] = {"file_name": os.path.basename(src_path) if src_path else ""}
    return record


def _epoch_of(doc_id):
    parsed = parse_document_id(doc_id)
    return derive_epoch(parsed[0]) if parsed else "unknown"


def main(args):
    record_files = sorted(glob.glob(os.path.join(PATH_DIR_RECORDS, "*", "*.json")))
    if not record_files:
        raise FileNotFoundError(
            f"no records under {PATH_DIR_RECORDS} (run scripts/indexing/build_index.py first)"
        )

    os.makedirs(os.path.dirname(PATH_FILE_RAG_CORPUS), exist_ok=True)

    by_epoch = Counter()
    by_unit  = Counter()
    written  = 0

    with open(PATH_FILE_RAG_CORPUS, "w", encoding="utf-8") as out:
        for path in record_files:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)

            record = _enrich(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            written += 1
            by_unit[record.get("unit_type", "?")] += 1
            by_epoch[_epoch_of(record.get("document_id", ""))] += 1

    print(f"[*] Wrote {written} records to {PATH_FILE_RAG_CORPUS}", flush=True)
    print(f"[*] per unit_type : {dict(by_unit)}", flush=True)
    print(f"[*] per epoch     : {dict(by_epoch)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidate indexed records -> one RAG corpus JSONL (outputs/rag/metadata/records.jsonl)"
    )
    main(parser.parse_args())
