# -*- coding: utf-8 -*-
"""
Bounding-box extraction from the U-Net noise channel

This module handles the extraction of spatial coordinates for noise instances by processing the binary noise mask. It uses aggressive morphological dilation to consolidate fragmented ink strokes into contiguous blobs, ensuring that historical marks are captured as single entities for subsequent classification
"""

import os
import cv2
import numpy as np
from typing import List
from src.core.config import SEG_THRESHOLDS

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_PATH, "data")
STORAGE_DIR = os.path.join(BASE_PATH, "storage")

KERNEL_DILATION = (7, 7)
AREA_MIN_BLOB = 200
THRESHOLD_PROB = SEG_THRESHOLDS[0]

def extract_noise_bboxes(mask_or_prob: np.ndarray, threshold: float = THRESHOLD_PROB) -> List[List[int]]:
    """
    Extract spatial bounding boxes from the noise prediction channel

    Args:
        mask_or_prob (np.ndarray): Probability map or binary mask of noise channel
        threshold (float): Binarization threshold for probability maps
    Returns:
        List[List[int]]: Collection of xywh bounding boxes sorted by reading order
    """
    if mask_or_prob.dtype != np.uint8:
        binary = (mask_or_prob > threshold).astype(np.uint8) * 255
    else:
        binary = np.where(mask_or_prob > 0, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_DILATION)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in contours:
        if cv2.contourArea(c) < AREA_MIN_BLOB:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append([int(x), int(y), int(w), int(h)])
    return sorted(bboxes, key=lambda b: (b[1], b[0]))  # Sort by top-to-bottom then left-to-right
