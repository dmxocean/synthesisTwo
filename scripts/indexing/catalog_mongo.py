# -*- coding: utf-8 -*-
"""
MongoDB catalog ingestion utility for archival records

This module populates a MongoDB instance with structured document records from the consolidated RAG corpus. It reads the JSONL metadata manifest and performs upsert operations to maintain a queryable catalog that mirrors the filesystem-based record store. This provides a high-performance alternative for document discovery and metadata retrieval within the RADAR ecosystem
"""

import os
import json
import argparse

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexing.stores import MongoWriter

PATH_FILE_RAG_CORPUS = os.path.join(BASE_PATH, "outputs", "rag", "metadata", "records.jsonl")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "RADAR")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "records")

def _read_jsonl(path):
    """
    Read and parse a line-delimited JSON file
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main(args):
    """
    Execute the MongoDB catalog population process
    """
    if not os.path.exists(PATH_FILE_RAG_CORPUS):
        raise FileNotFoundError(
            f"corpus not found at {PATH_FILE_RAG_CORPUS} (run scripts/rag/ingest_real.py first)"
        )

    records = _read_jsonl(PATH_FILE_RAG_CORPUS)  # Load consolidated records manifest
    writer = MongoWriter(args.mongo_uri, db=args.db, collection=args.collection)
    n = writer.upsert_dicts(records)  # Perform bulk upsert into the catalog

    print(f"[*] Upserted {n} records into {args.db}.{args.collection} at {args.mongo_uri}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upsert the RAG corpus JSONL into a MongoDB catalog")
    parser.add_argument("--mongo-uri", default=MONGO_URI, help="MongoDB connection URI")
    parser.add_argument("--db", default=MONGO_DB, help="Database name")
    parser.add_argument("--collection", default=MONGO_COLLECTION, help="Collection name")
    args = parser.parse_args()
    main(args)
