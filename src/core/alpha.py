# -*- coding: utf-8 -*-
"""
Asset alpha refinement and content-driven polygon extraction

This module implements algorithms for separating ink and stamp pixels from paper background bleed-through in manually cropped assets. It provides adaptive thresholding based on HSV value and saturation to ensure high-precision separation of historical ink. Additionally, it exposes utilities for deriving pixel-accurate polygons from rendered RGBA layers for ground-truth generation
"""

import os
from typing import List, Optional, Any

import numpy as np
import cv2
from PIL import Image

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_PATH, "data")
STORAGE_DIR = os.path.join(BASE_PATH, "storage")

V_FLOOR_DARK = 80
V_CEILING_OTSU = 180
FLOOR_SATURATION = 50
SIZE_KERNEL_CLOSE = 5
SIZE_KERNEL_LINE = (15, 3)
SIZE_KERNEL_TEXT = (35, 5)
SIZE_KERNEL_TIGHT = 5
ASPECT_LINE_THRESHOLD = 5.0
AREA_MIN_BLOB = 4
AREA_MIN_POLYGON_NS = 20
AREA_MIN_POLYGON_TEXT = 50
EPS_SIMPLIFY_TIGHT = 1.5
EPS_SIMPLIFY_TEXT = 2.5
RATIO_FALLBACK_GUARD = 0.10

def clean_real_asset_alpha(rgba: Image.Image) -> Image.Image:
    """
    Refine an RGBA asset alpha to isolate ink from paper background

    The function uses gamma correction and morphological black-hat transforms followed by a biased Otsu threshold to ensure zero paper background survives in the alpha channel
    """
    arr = np.array(rgba)
    bgr = arr[..., :3]
    original_alpha = arr[..., 3]
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)

    gamma = 0.8
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    darkened = cv2.LUT(gray, table)

    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(darkened, cv2.MORPH_BLACKHAT, kernel_bh)

    mask_visible = original_alpha > 0
    if not mask_visible.any():
        return rgba
        
    pixels_to_threshold = blackhat[mask_visible]
    if len(pixels_to_threshold) < 10:
        return rgba

    t_val, _ = cv2.threshold(pixels_to_threshold, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    t_aggressive = min(255, t_val + 15)
    _, mask = cv2.threshold(blackhat, t_aggressive, 255, cv2.THRESH_BINARY)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    final_alpha = cv2.bitwise_and(mask_closed, original_alpha)
    
    num, labels, stats, _ = cv2.connectedComponentsWithStats(final_alpha, connectivity=8)
    if num > 1:
        largest_components = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= 10]
        final_alpha = np.isin(labels, largest_components).astype(np.uint8) * 255
        
    cleaned_rgba = np.dstack((bgr, final_alpha))
    return Image.fromarray(cleaned_rgba, mode="RGBA")

def load_manual_asset(path_abs: str) -> Optional[Image.Image]:
    """
    Load a manually cropped asset from the filesystem as an RGBA image
    """
    try:
        return Image.open(path_abs).convert("RGBA")
    except Exception:
        return None

def content_mask_from_layer(layer: Image.Image, template_polygon: Optional[List[int]], 
                            width: int, height: int) -> np.ndarray:
    """
    Build a boolean content mask from a rendered RGBA layer

    Args:
        layer (Image.Image): RGBA canvas containing the rendered region
        template_polygon (Optional[List[int]]): Polygon defining the region boundary
        width (int): Width of the target canvas
        height (int): Height of the target canvas
    Returns:
        np.ndarray: Boolean array where True indicates the presence of rendered content
    """
    return np.array(layer)[..., 3] > 0

def polygon_from_content_tight(content_mask: np.ndarray) -> List[List[int]]:
    """
    Extract pixel-tight polygons from a noise content mask
    """
    mask_u8 = content_mask.astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        return []

    kernel_iso = np.ones((SIZE_KERNEL_TIGHT, SIZE_KERNEL_TIGHT), np.uint8)
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_iso)

    contours_probe, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_probe:
        x, y, bw, bh = cv2.boundingRect(c)
        long_side = max(bw, bh)
        short_side = max(1, min(bw, bh))
        if long_side / short_side >= ASPECT_LINE_THRESHOLD:
            if bw >= bh:
                kernel_dir = cv2.getStructuringElement(cv2.MORPH_RECT, SIZE_KERNEL_LINE)
            else:
                kh, kw = SIZE_KERNEL_LINE
                kernel_dir = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            roi = closed[y:y + bh, x:x + bw]
            closed[y:y + bh, x:x + bw] = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel_dir)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < AREA_MIN_POLYGON_NS:
            continue
        simplified = cv2.approxPolyDP(c, EPS_SIMPLIFY_TIGHT, closed=True)
        if len(simplified) < 3:
            continue
        flat = simplified.reshape(-1, 2).flatten().tolist()
        polys.append([int(v) for v in flat])
    return polys

def polygon_from_content_blobs(content_mask: np.ndarray) -> List[List[int]]:
    """
    Extract per-blob polygons from a text content mask
    """
    mask_u8 = content_mask.astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, SIZE_KERNEL_TEXT)
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < AREA_MIN_POLYGON_TEXT:
            continue
        simplified = cv2.approxPolyDP(c, EPS_SIMPLIFY_TEXT, closed=True)
        if len(simplified) < 3:
            continue
        flat = simplified.reshape(-1, 2).flatten().tolist()
        polys.append([int(v) for v in flat])
    return polys
