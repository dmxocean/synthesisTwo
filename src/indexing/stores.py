# -*- coding: utf-8 -*-
"""
Persistence for the offline indexing run (components)

Three derived stores back the app (see the plan): the record JSON files, the
MongoDB catalog, and the blob/derived images

This module writes the records to disk (always) and, optionally, upserts them into MongoDB

The Qdrant vector index is built in the rag stage (its env has the embedding
model), not here, so the GPU indexing env stays free of llama-index/qdrant deps

  write_records_json : Record[] -> data/index/records/{doc}/{page}.{unit}.json
  MongoWriter        : optional catalog upsert (lazy pymongo import)
"""

import os
import json

from src.core.config import MONGO_DB, MONGO_COLLECTION

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write_records_json(records, out_dir):
    """
    Write one JSON file per record including page and region records

    Args:
        records (List[Record]): pydantic records from detection/records/builder
        out_dir (str): destination directory
    Returns:
        int: number of files written
    """
    os.makedirs(out_dir, exist_ok=True)
    for rec in records:
        suffix   = "page" if rec.unit_type == "page" else (rec.region_id or "region")
        out_path = os.path.join(out_dir, f"{rec.page_id}.{suffix}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec.model_dump(), f, ensure_ascii=False, indent=2)
    return len(records)


class MongoWriter:
    """
    Optional MongoDB catalog upsert with lazy pymongo import

    One document per record, keyed by record_id, in the configured collection
    """

    def __init__(self, uri="mongodb://localhost:27017", db=MONGO_DB, collection=MONGO_COLLECTION):
        from pymongo import MongoClient   # Lazy: only needed when Mongo is used
        self.coll = MongoClient(uri)[db][collection]

    def upsert(self, records):
        """
        Upsert pydantic Records by record_id and return the count
        """
        return self.upsert_dicts(rec.model_dump() for rec in records)

    def upsert_dicts(self, records):
        """
        Upsert pre-serialised record dicts by record_id
        """
        n = 0
        for doc in records:
            self.coll.replace_one({"record_id": doc["record_id"]}, doc, upsert=True)
            n += 1
        return n
