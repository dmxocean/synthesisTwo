# -*- coding: utf-8 -*-
"""
Archival metadata derivation from canonical document identifiers

This module provides utilities for extracting archival metadata fields from the GUIRAD document identifier format. It derives year, month, and day information to generate human-readable titles, ISO dates, and historical epoch mappings. These fields are used to enrich the records during both indexing and RAG ingestion phases
"""

import os
import re
from typing import Optional, Tuple, Dict, Any

from src.core.config import DICT_EPOCHS

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_PATH, "data")
STORAGE_DIR = os.path.join(BASE_PATH, "storage")

DOC_RE = re.compile(r"a(\d{4})m(\d{1,2})(?:d(\d{1,2}))?")

MONTHS_ES = [
    "", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]

COLLECTION_ID = "uab_radio_scripts"
COLLECTION_NAME = "UAB Radio Scripts Archive"
INSTITUTION = "Universitat Autònoma de Barcelona"
REPOSITORY = "UAB Library"
FONDS = "Radio Barcelona"
SERIES = "Guiones de emisión"

def parse_document_id(doc_id: str) -> Optional[Tuple[int, int, Optional[int]]]:
    """
    Parse a GUIRAD document identifier into year, month, and optional day
    """
    m = DOC_RE.search(doc_id or "")
    if not m:
        return None
    year = int(m.group(1))
    month = int(m.group(2))
    day = int(m.group(3)) if m.group(3) else None
    return year, month, day

def derive_epoch(year: int) -> str:
    """
    Map a specific year to its corresponding historical epoch name
    """
    for name, year_range in DICT_EPOCHS.items():
        if year in year_range:
            return name
    return "unknown"

def derive_title(doc_id: str) -> str:
    """
    Generate a human-readable title from a document identifier
    """
    parsed = parse_document_id(doc_id)
    if not parsed:
        return doc_id or ""
    year, month, _ = parsed
    month_name = MONTHS_ES[month] if 1 <= month <= 12 else str(month)
    return f"Radio Barcelona - {month_name} {year}"

def derive_date(doc_id: str) -> Optional[str]:
    """
    Derive an ISO formatted date string from a document identifier
    """
    parsed = parse_document_id(doc_id)
    if not parsed:
        return None
    year, month, day = parsed
    return f"{year:04d}-{month:02d}-{(day or 1):02d}"

def derive_archival(doc_id: str) -> Dict[str, Any]:
    """
    Generate a complete archival metadata block for a document
    """
    parsed = parse_document_id(doc_id)
    year = parsed[0] if parsed else None
    return {
        "collection_id": COLLECTION_ID,
        "collection_name": COLLECTION_NAME,
        "archival_metadata": {
            "fonds": FONDS,
            "series": SERIES,
            "epoch": derive_epoch(year) if year is not None else "unknown",
            "call_number": doc_id,
            "institution": INSTITUTION,
            "repository": REPOSITORY,
        },
    }
