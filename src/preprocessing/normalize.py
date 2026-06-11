# -*- coding: utf-8 -*-
"""
Fast Morphological Background Normalisation + Ink Mask Extraction

Replaces the legacy adaptiveThreshold and PDE inpainting with a blazing-fast
Morphological Black-Hat and Biased Otsu pipeline. 
This is used at inference time (predict.py) to generate a deterministic ink_mask
that prevents the models from hallucinating text on blank paper margins.

Pipeline role:
  Inputs  RGB page (H, W, 3) uint8
  Outputs (normalised RGB (H, W, 3) uint8, ink_mask (H, W) bool)
"""

import cv2
import numpy as np
import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _aggressive_ink_mask_and_paper(img_rgb):
    """
    Return the aggressive ink mask and the morphologically closed paper estimate
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Morphological Black-Hat
    # Kernel size 25x25 covers the thickest pen strokes
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
    
    # 2. Aggressive Biased Otsu
    # Otsu calculates the optimal threshold T
    T, _ = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Add a bias to guarantee no background noise survives
    T_aggressive = min(255, T + 10) 
    _, mask = cv2.threshold(blackhat, T_aggressive, 255, cv2.THRESH_BINARY)
    
    # 3. Fast Paper Generation (Morphological Close)
    # Instead of inpainting, we smear the bright paper pixels over the dark ink
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    paper = cv2.morphologyEx(img_rgb, cv2.MORPH_CLOSE, kernel_close)
    
    return mask, paper


def normalise_background(img_rgb):
    """
    Neutralise paper and return the cleaned image plus the ink mask

    Args:
        img_rgb (np.ndarray): (H, W, 3) uint8 RGB page
    Returns:
        Tuple[np.ndarray, np.ndarray]: normalised RGB image and boolean ink mask
    """
    mask, paper = _aggressive_ink_mask_and_paper(img_rgb)
    
    # Per-pixel deviation calculation
    delta       = cv2.subtract(paper, img_rgb)
    normalised  = cv2.bitwise_not(delta)
    ink_mask    = (mask > 0)
    
    return normalised, ink_mask


def preprocess_blackhat(img_rgb):
    """
    Direct inverted Black-Hat enhanced inference path
    """
    g  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    k  = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    return cv2.cvtColor(255 - bh, cv2.COLOR_GRAY2RGB)


def _apply_contrast(img_rgb, gamma):
    """
    Gamma LUT contrast boost where gamma > 1 darkens ink and brightens paper
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img_rgb, table)


def _apply_bg_deletion(img_rgb, sigma):
    """
    Large-sigma Gaussian division to flatten uneven background illumination
    """
    img_f = img_rgb.astype(np.float32) / 255.0
    blur  = cv2.GaussianBlur(img_f, (0, 0), sigma)
    norm  = img_f / (blur + 1e-6)
    return (np.clip(norm, 0, 1) * 255).astype(np.uint8)


def _apply_gaussian(img_rgb, sigma):
    """
    Gaussian blur for light denoising after contrast and background removal
    """
    return cv2.GaussianBlur(img_rgb, (0, 0), sigma)


def preprocess_ctr_bgd_gs(img_rgb):
    """
    Full production pipeline including normalisation, contrast, and denoising

    Returns:
        Tuple[np.ndarray, np.ndarray]: preprocessed image and boolean ink mask
    """
    norm, ink_mask = normalise_background(img_rgb)
    img = _apply_contrast(norm, 2.5)
    img = _apply_bg_deletion(img, 100)
    img = _apply_gaussian(img, 1.0)
    return img, ink_mask


def extract_ink_on_white(page_rgb, binary_mask):
    """
    Return original ink pixels on white canvas for VLM input

    IMPORTANT: page_rgb must be the unprocessed original page to preserve ink colour

    Args:
        page_rgb (np.ndarray): (H, W, 3) uint8 original RGB page
        binary_mask (np.ndarray): (H, W) bool where True represents ink to keep
    Returns:
        np.ndarray: (H, W, 3) uint8 with original ink on white background
    """
    canvas = np.full_like(page_rgb, 255)
    canvas[binary_mask] = page_rgb[binary_mask]
    return canvas
