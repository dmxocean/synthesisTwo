# -*- coding: utf-8 -*-
"""
Metadata builder: fuses model outputs into v2 RAG-compatible records

Takes the per-page predictions from the segmentation model (SegFormer or U-Net),
Qwen3-VL transcription, and ResNet-18 noise classifier and assembles Record
objects validated against the pydantic schema in schema.py.

Pipeline role:
  Inputs  seg_pred (npy), ocr_pred, htr_pred, noise_pred (all from predict.py)
  Outputs list[Record] -> serialised to JSON by pipeline/run.py

Critical design decisions:
  Chunking: one page record (unit_type=page) always, plus region records for
  connected printed components > AREA_MIN_REGION_FRAC of page and for
  handwritten line clusters with vertical gap < GAP_CLUSTER_PX.
  Noise objects are embedded in parent region visual_marks[], not standalone.
  Overlap alerts are derived by comparing noise bboxes against text region bboxes
"""

import os
import math
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from src.detection.records.schema import (
    Record, OcrPayload, VisualMark, LinkedSpan, ForensicFlags, Provenance
)
from src.core.archival import derive_title, derive_date

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

AREA_MIN_REGION_FRAC = 0.01   # Printed regions smaller than 1 % of page area are skipped
GAP_CLUSTER_PX       = 40     # Vertical gap above which handwritten lines become separate records
IOU_OVERLAP_ALERT    = 0.05   # Noise/text IoU threshold for stamp_over_text alert
IOU_STRIKETHROUGH    = 0.30   # Noise/text IoU threshold for strikethrough alert
IOU_LINKED_SPAN      = 0.10   # Noise/text IoU threshold for linked_text_spans entry
CONF_VERIFIED        = 0.85   # Minimum overall confidence for verification_status = verified


def _iou(bbox_a, bbox_b):
    """
    Intersection-over-union for two [x, y, w, h] bboxes
    """
    ax0, ay0, aw, ah = bbox_a
    bx0, by0, bw, bh = bbox_b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / max(1, union)


def _overlap_alerts(noise_instances, text_bboxes):
    """
    Return alert strings and linked span entries for overlapping noise/text pairs

    Args:
        noise_instances (list): NoiseInstance objects
        text_bboxes (list): [[x, y, w, h], ...] bboxes of text regions
    Returns:
        Tuple[List[str], List[LinkedSpan]]: alerts and linked spans
    """
    alerts = []
    spans  = []
    for ni in noise_instances:
        for tid, tbbox in enumerate(text_bboxes):
            iou_val = _iou(ni.bbox, tbbox)
            if iou_val > IOU_OVERLAP_ALERT and ni.mark_type in ("stamps",):
                if "stamp_over_text" not in alerts:
                    alerts.append("stamp_over_text")
            if iou_val > IOU_STRIKETHROUGH and ni.mark_type in ("lines", "crosses"):
                if "strikethrough_text" not in alerts:
                    alerts.append("strikethrough_text")
            if iou_val > IOU_LINKED_SPAN:
                spans.append(LinkedSpan(
                    span_id   = f"span_{ni.mark_id}_r{tid:03d}",
                    mark_id   = ni.mark_id,
                    region_id = f"r{tid:03d}",
                    iou       = round(iou_val, 4),
                ))
    return alerts, spans


def _build_provenance(image_path, pipeline_version=""):
    """
    Return a Provenance object for the current pipeline run
    """
    return Provenance(
        pipeline_version  = pipeline_version,
        model_outputs     = {},
        created_at        = datetime.now(timezone.utc).isoformat(),
        source_image_path = image_path,
    )


def build_records(
    document_id:    str,
    page_id:        str,
    image_path:     str,
    seg_pred:      np.ndarray,     # (3, H, W) float probabilities
    ocr_pred:       list,           # list[OcrRegion]
    htr_pred:       list,           # list[HtrRegion]
    noise_pred:     list,           # list[NoiseInstance]
    pipeline_version: str = "",
) -> List[Record]:
    """
    Assemble per-page and per-region Record objects

    Args:
        document_id (str): unique document identifier
        page_id (str): unique page identifier within the document
        image_path (str): absolute path to the source page PNG
        seg_pred (np.ndarray): (3, H, W) sigmoid probability map from U-Net
        ocr_pred (list): printed-region results from TrOCR-Printed
        htr_pred (list): handwritten-region results from TrOCR-Handwritten
        noise_pred (list): noise instance results from ResNet-18
        pipeline_version (str): git SHA or version tag for traceability
    Returns:
        List[Record]: one page record and N region records, all pydantic-validated
    """
    H, W = seg_pred.shape[1], seg_pred.shape[2]
    page_area = H * W

    prov = _build_provenance(image_path, pipeline_version)

    # Archival fields derived once from the document_id (shared with the RAG bridge)
    doc_title = derive_title(document_id)
    doc_date  = derive_date(document_id)

    # Page-level record aggregation
    all_ocr_text  = " ".join(r.full_text for r in ocr_pred if r.full_text)
    all_htr_text  = " ".join(r.full_text for r in htr_pred if r.full_text)
    all_mark_types = [ni.mark_type for ni in noise_pred]

    all_text_bboxes = [r.bbox for r in ocr_pred] + [r.bbox for r in htr_pred]
    page_alerts, page_spans = _overlap_alerts(noise_pred, all_text_bboxes)

    # Confidence calculation via median of region scores
    all_confs = ([r.confidence_score for r in ocr_pred] +
                 [r.confidence_score for r in htr_pred] +
                 [ni.confidence_score for ni in noise_pred])
    page_conf = float(np.median(all_confs)) if all_confs else 0.0

    if "hw ∧ pr overlap" in page_alerts or any(
        _iou(o.bbox, h.bbox) > 0.10 for o in ocr_pred for h in htr_pred
    ):
        if "handwritten_annotation_on_printed" not in page_alerts:
            page_alerts.append("handwritten_annotation_on_printed")

    all_mark_descriptions = " ".join(
        ni.description for ni in noise_pred if ni.description
    )

    page_retrieval = " | ".join(filter(None, [
        all_ocr_text, all_htr_text,
        ", ".join(all_mark_types),
        all_mark_descriptions,
        ", ".join(page_alerts),
    ]))

    visual_marks_page = [
        VisualMark(
            mark_id          = ni.mark_id,
            mark_type        = ni.mark_type,
            bbox             = ni.bbox,
            confidence_score = ni.confidence_score,
            description      = ni.description,
        )
        for ni in noise_pred
    ]

    page_record = Record(
        record_id         = f"{document_id}__{page_id}__page",
        document_id       = document_id,
        page_id           = page_id,
        region_id         = None,
        unit_type         = "page",
        title             = doc_title,
        date_created      = doc_date,
        retrieval_text    = page_retrieval,
        ocr               = OcrPayload(
            full_text        = f"{all_ocr_text} {all_htr_text}".strip(),
            language         = "es",
            engine           = "pipeline",
            confidence_score = page_conf,
        ),
        visual_marks      = visual_marks_page,
        linked_text_spans = page_spans,
        forensic_flags    = ForensicFlags(
            verification_status   = "verified" if page_conf >= CONF_VERIFIED else "uncertain",
            alerts                = page_alerts,
            human_review_required = any(
                a in page_alerts for a in ("strikethrough_text", "handwritten_annotation_on_printed")
            ),
            confidence_score      = page_conf,
        ),
        provenance = prov,
    )

    records = [page_record]

    # Region-level record processing
    for rid, region in enumerate(ocr_pred):
        if region.bbox[2] * region.bbox[3] < AREA_MIN_REGION_FRAC * page_area:
            continue
        noise_in_region = [ni for ni in noise_pred if _iou(ni.bbox, region.bbox) > IOU_LINKED_SPAN]
        r_alerts, r_spans = _overlap_alerts(noise_in_region, [region.bbox])
        records.append(Record(
            record_id         = f"{document_id}__{page_id}__pr_{rid:04d}",
            document_id       = document_id,
            page_id           = page_id,
            region_id         = region.region_id,
            unit_type         = "region",
            title             = doc_title,
            date_created      = doc_date,
            bbox              = region.bbox,
            reading_order     = region.reading_order,
            retrieval_text    = region.full_text,
            ocr               = OcrPayload(
                full_text        = region.full_text,
                language         = "es",
                engine           = region.engine,
                confidence_score = region.confidence_score,
                word_confidences = region.word_confidences,
            ),
            visual_marks      = [
                VisualMark(mark_id=ni.mark_id, mark_type=ni.mark_type, bbox=ni.bbox,
                           confidence_score=ni.confidence_score)
                for ni in noise_in_region
            ],
            linked_text_spans = r_spans,
            forensic_flags    = ForensicFlags(
                verification_status   = "verified" if region.confidence_score >= CONF_VERIFIED else "uncertain",
                alerts                = r_alerts,
                human_review_required = len(r_alerts) > 0,
                confidence_score      = region.confidence_score,
            ),
            provenance = prov,
        ))

    for rid, region in enumerate(htr_pred):
        if region.bbox[2] * region.bbox[3] < AREA_MIN_REGION_FRAC * page_area:
            continue
        noise_in_region = [ni for ni in noise_pred if _iou(ni.bbox, region.bbox) > IOU_LINKED_SPAN]
        r_alerts, r_spans = _overlap_alerts(noise_in_region, [region.bbox])
        records.append(Record(
            record_id         = f"{document_id}__{page_id}__hw_{rid:04d}",
            document_id       = document_id,
            page_id           = page_id,
            region_id         = region.region_id,
            unit_type         = "region",
            title             = doc_title,
            date_created      = doc_date,
            bbox              = region.bbox,
            reading_order     = region.reading_order,
            retrieval_text    = region.full_text,
            ocr               = OcrPayload(
                full_text        = region.full_text,
                language         = "es",
                engine           = region.engine,
                confidence_score = region.confidence_score,
                word_confidences = region.word_confidences,
            ),
            visual_marks      = [
                VisualMark(mark_id=ni.mark_id, mark_type=ni.mark_type, bbox=ni.bbox,
                           confidence_score=ni.confidence_score)
                for ni in noise_in_region
            ],
            linked_text_spans = r_spans,
            forensic_flags    = ForensicFlags(
                verification_status   = "verified" if region.confidence_score >= CONF_VERIFIED else "uncertain",
                alerts                = r_alerts,
                human_review_required = len(r_alerts) > 0,
                confidence_score      = region.confidence_score,
            ),
            provenance = prov,
        ))

    return records
