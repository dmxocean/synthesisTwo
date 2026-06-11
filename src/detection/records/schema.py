# -*- coding: utf-8 -*-
"""
Pydantic v2 models matching the rag/llm_rag_pipeline_v2.py record schema

These dataclasses define every field expected by RadioArchiveMetadataReader
so the metadata builder can validate its output before writing JSON and so
the CLI tools can deserialise records without implicit key guessing

Pipeline role:
  Imported by metadata/builder.py and rag/ingest_real.py
  Validated records are serialised to JSON via model_dump()
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

import os
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class WordConfidence(BaseModel):
    """
    One transcribed word with its token-logprob confidence for word-level colouring
    """
    text:       str   = ""
    confidence: float = 0.0


class OcrPayload(BaseModel):
    """
    OCR or HTR transcription result for one region
    """
    full_text:        str                   = ""
    keywords:         List[str]             = Field(default_factory=list)
    language:         str                   = "es"
    engine:           str                   = ""
    confidence_score: float                 = 0.0
    word_confidences: List[WordConfidence]  = Field(default_factory=list)


class SummaryPayload(BaseModel):
    """
    High-level summary fields
    """
    short_description: str       = ""
    keywords:          List[str] = Field(default_factory=list)


class EntityMention(BaseModel):
    """
    Named entity extracted from the OCR/HTR text
    """
    entity_type: str   = ""
    text:        str   = ""
    count:       int   = 1


class VisualMark(BaseModel):
    """
    One detected noise instance including stamps, circles, crosses, lines, and marks
    """
    mark_id:          str
    mark_type:        str             # One of: stamps, circles, crosses, lines, marks
    subtype:          Optional[str]   = None
    description:      str             = ""
    bbox:             List[int]       = Field(default_factory=list)   # [x, y, w, h]
    confidence_score: float           = 0.0
    verification_status: str          = "uncertain"


class LinkedSpan(BaseModel):
    """
    Link between a noise mark and an overlapping text span
    """
    span_id:      str
    mark_id:      str
    region_id:    str
    text:         str   = ""
    iou:          float = 0.0


class ForensicFlags(BaseModel):
    """
    Quality and verification signals for one record
    """
    verification_status:    str            = "uncertain"
    alerts:                 List[str]      = Field(default_factory=list)
    human_review_required:  bool           = False
    confidence_score:       float          = 0.0


class Provenance(BaseModel):
    """
    Traceability metadata for the pipeline run that produced this record
    """
    pipeline_version:   str                = ""
    model_outputs:      Dict[str, Any]     = Field(default_factory=dict)
    created_at:         str                = ""
    source_image_path:  str                = ""


class Record(BaseModel):
    """
    Complete metadata record matching the v2 RadioArchiveMetadataReader schema
    """
    record_id:          str
    document_id:        str
    page_id:            str
    region_id:          Optional[str]                   = None
    unit_type:          Literal["page", "region"]       = "page"
    bbox:               List[int]                        = Field(default_factory=list)   # [x, y, w, h] of the text region on the page
    reading_order:      int                              = 0                              # Column-aware reading-order index within the page
    source_type:        str                              = "radio_script"
    title:              str                              = ""
    date_created:       Optional[str]                   = None
    language:           List[str]                        = Field(default_factory=lambda: ["es"])
    retrieval_text:     str                              = ""
    ocr:                OcrPayload                       = Field(default_factory=OcrPayload)
    summary:            Optional[SummaryPayload]         = None
    entities:           List[EntityMention]              = Field(default_factory=list)
    visual_marks:       List[VisualMark]                 = Field(default_factory=list)
    linked_text_spans:  List[LinkedSpan]                 = Field(default_factory=list)
    forensic_flags:     ForensicFlags                    = Field(default_factory=ForensicFlags)
    provenance:         Provenance                       = Field(default_factory=Provenance)
