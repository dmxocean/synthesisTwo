# -*- coding: utf-8 -*-
"""
Authentic paper-texture recovery components

Reusable pieces for recovering clean A4 paper from raw GUIRAD PDFs by erasing
ink. The harvesting workflow (year discovery, task planning, parallel dispatch)
lives in scripts/preprocessing/recover_paper.py - this module only provides:

  - get_epoch                   : map a year to its historical epoch
  - clean_ink_and_recover_paper : the ink-removal algorithm (morphological close)
  - process_paper_task          : the per-PDF worker (kept importable so the
                                  ProcessPoolExecutor in the script can pickle it)
"""

import os
import cv2
import numpy as np

from src.preprocessing.pdf import DocumentProcessor
from src.core.config import DICT_EPOCHS

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_epoch(year):
    """
    Return the historical epoch name for a given year, or None if out of range
    """
    for name, year_range in DICT_EPOCHS.items():
        if int(year) in year_range:
            return name
    return None


def clean_ink_and_recover_paper(img_rgb, saturation_scale=0.25):
    """
    Erase ink to recover the underlying paper texture

    Uses a fast morphological close instead of slow inpainting. A 35x35
    kernel bridges most handwriting and printed gaps

    After the close, saturation is suppressed so that residual chroma from
    colored ink is neutralised toward the neutral paper tone

    Args:
        img_rgb (np.ndarray): (H, W, 3) RGB page
        saturation_scale (float): multiplier applied to the HSV S channel (0-1)
    Returns:
        np.ndarray: (H, W, 3) RGB paper texture with ink removed
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    cleaned = cv2.morphologyEx(img_rgb, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)

    hsv = cv2.cvtColor(cleaned, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Only suppress pixels whose saturation is clearly above the paper's natural warmth
    ink_mask = hsv[:, :, 1] > 50
    hsv[:, :, 1][ink_mask] *= saturation_scale
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def process_paper_task(task):
    """
    Worker: recover paper from a single PDF and write one PNG

    Args:
        task (tuple): (pdf_path, epoch_name, year, count_idx, output_dir)
    Returns:
        str|None: path on success, None on failure
    """
    pdf_path, epoch_name, year, count_idx, output_dir = task
    try:
        a4_standard = DocumentProcessor.pdf_to_standard_image(pdf_path, dpi=300)
        if a4_standard is None:
            return None

        clean_paper = clean_ink_and_recover_paper(a4_standard)
        target_path = os.path.join(output_dir, f"clean_paper_{year}_{count_idx:03d}.png")
        cv2.imwrite(target_path, cv2.cvtColor(clean_paper, cv2.COLOR_RGB2BGR))
        return f"{epoch_name}/{os.path.basename(target_path)}"
    except Exception:
        return None
