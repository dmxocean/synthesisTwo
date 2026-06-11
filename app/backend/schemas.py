# -*- coding: utf-8 -*-
"""
API response models for the viewer (components)

Shapes the persisted records and derived-image paths into the compact JSON the
frontend renders

These are read models only; the canonical record schema lives in
src/detection/records/schema.py
"""

import os
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MarkView(BaseModel):
    """A typed noise mark with its location and confidence for overlay rendering"""
    mark_id:    str
    mark_type:  str
    bbox:       List[int] = Field(default_factory=list)  # [x, y, w, h]
    confidence: float = 0.0


class RegionView(BaseModel):
    """One transcribed region (printed or handwritten) with its position + confidence"""
    region_id:           str
    layer:               str               # "printed" or "handwritten"
    text:                str = ""
    confidence:          float = 0.0
    verification_status: str = "uncertain"
    bbox:                List[int] = Field(default_factory=list)  # [x, y, w, h] for positioned overlay
    reading_order:       int = 0                                  # Column-aware order on the page
    word_confidences:    List[Dict] = Field(default_factory=list) # [{text, confidence}] for word colouring


class DocumentSummary(BaseModel):
    """One document in the catalog listing"""
    document_id: str
    n_pages:     int
    pages:       List[str] = Field(default_factory=list)


class PageDetail(BaseModel):
    """Everything the viewer needs for one page"""
    document_id:         str
    page_id:             str
    title:               str = ""
    date_created:        Optional[str] = None
    language:            List[str] = Field(default_factory=list)
    printed_text:        str = ""
    handwritten_text:    str = ""
    printed_regions:     List[RegionView] = Field(default_factory=list)
    handwritten_regions: List[RegionView] = Field(default_factory=list)
    marks:               List[MarkView] = Field(default_factory=list)
    forensic:            Dict = Field(default_factory=dict)
    confidence:          float = 0.0
    layers:              Dict[str, str] = Field(default_factory=dict)  # Name -> derived image url
