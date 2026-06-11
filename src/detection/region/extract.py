# -*- coding: utf-8 -*-
"""
Region extraction from a segmentation layer (component)

Turns one layer's probability/mask channel (handwritten or printed) into a list
of region bounding boxes using the classic semantic-segmentation -> connected-
components post-processing (dhSegment / Doc-UFCN style): RLSA-like morphological
smoothing merges glyphs into text lines/blocks, then connected components with a
minimum-area filter, returned in reading order. These regions are cropped and
read by the VLM (detection/vlm) and assembled by detection/records/builder.

Generalizes the noise bbox extractor (detection/noise/bbox) from the noise channel
to the text channels, with wider horizontal smoothing suited to text lines.
"""

import os
import dataclasses
from typing import List

import cv2
import numpy as np

from src.core.config import SEG_THRESHOLDS as _SEG_THRESHOLDS

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

THRESHOLD_PROB  = _SEG_THRESHOLDS[2]  # PR channel (index 2) - used as generic default
KERNEL_RLSA     = (25, 9)   # (w, h) wide horizontal smoothing to bridge inter-word gaps into lines
AREA_MIN_REGION = 400       # Drop specks below this pixel area


@dataclasses.dataclass
class RegionBox:
    """
    A proposed region: id + bbox + reading order
    """
    region_id: str
    bbox:      List[int]    # [x, y, w, h] in page pixel coordinates
    order:     int = 0      # Column-aware reading-order index on the page


def reading_order(boxes, line_tol=None, col_gap=None):
    """
    Sort bboxes in column-aware reading order

    Pure y-banding interleaves side-by-side columns at the same height; instead we
    split the boxes into columns by gaps between their x-centres, order the columns
    left-to-right, then read each column top-to-bottom. Deterministic and explainable
    (no model) - the right default for an archival tool that must justify its ordering
    
    Args:
        boxes (list): [[x, y, w, h], ...]
        line_tol (int): vertical tolerance for "same line"; defaults to median box height
        col_gap (int): min x-centre gap that starts a new column; defaults to median box width
    Returns:
        list: the same bboxes, reordered
    """
    if not boxes:
        return boxes
    if line_tol is None:
        line_tol = max(1, int(np.median([h for _, _, _, h in boxes])))
    if col_gap is None:
        col_gap = max(int(np.median([w for _, _, w, _ in boxes])), line_tol * 3)

    centres = sorted((x + w / 2.0) for x, _, w, _ in boxes)
    bounds = [(centres[i - 1] + centres[i]) / 2.0
              for i in range(1, len(centres)) if centres[i] - centres[i - 1] > col_gap]

    def _column(b):
        c = b[0] + b[2] / 2.0
        return sum(1 for bd in bounds if c > bd)

    return sorted(boxes, key=lambda b: (_column(b), b[1] // line_tol, b[0]))


def extract_regions(mask_or_prob, prefix="r", kernel=KERNEL_RLSA,
                    min_area=AREA_MIN_REGION, threshold=THRESHOLD_PROB):
    """
    Extract region bboxes from one layer mask or probability channel

    Args:
        mask_or_prob (np.ndarray): (H, W) float probabilities or binary uint8
        prefix (str): region_id prefix (e.g. 'pr' / 'hw')
        kernel (tuple): (w, h) RLSA smoothing kernel; wider w merges words into lines
        min_area (int): minimum contour area to keep
        threshold (float): binarisation threshold for float input
    Returns:
        List[RegionBox]: regions in reading order
    """
    if mask_or_prob.dtype != np.uint8:
        binary = (mask_or_prob > threshold).astype(np.uint8) * 255
    else:
        binary = np.where(mask_or_prob > 0, 255, 0).astype(np.uint8)

    k        = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([int(x), int(y), int(w), int(h)])

    boxes = reading_order(boxes)
    return [RegionBox(region_id=f"{prefix}_{i:04d}", bbox=b, order=i) for i, b in enumerate(boxes)]
