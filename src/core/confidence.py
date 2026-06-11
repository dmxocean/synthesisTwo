# -*- coding: utf-8 -*-
"""
Confidence and uncertainty utilities for archival processing

This module provides tools for converting raw model probabilities into calibrated confidence signals. It handles uncertainty mapping for segmentation, log-probability aggregation for transcription, and temperature scaling for classification. These utilities ensure that the system provides reliable indicators of prediction quality across different processing stages
"""

import os
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_PATH, "data")
STORAGE_DIR = os.path.join(BASE_PATH, "storage")

DEFAULT_UNKNOWN_TAU = 0.55
LABEL_UNKNOWN = "unknown"
WORD_SPACE_MARKERS = ("Ġ", "▁", " ")

def uncertainty_map(probs: np.ndarray, mode: str = "margin") -> np.ndarray:
    """
    Calculate a per-pixel uncertainty map in the range zero to one

    Args:
        probs (np.ndarray): Multi-channel class probabilities
        mode (str): Calculation method, either margin or entropy
    Returns:
        np.ndarray: Float uncertainty map where higher values indicate lower confidence
    """
    p = np.clip(probs.astype(np.float32), 1e-6, 1.0)
    if mode == "entropy":
        ent = -(p * np.log(p)).sum(axis=0)
        return (ent / math.log(max(2, p.shape[0]))).astype(np.float32)
    return (1.0 - p.max(axis=0)).astype(np.float32)

def region_confidence(prob_channel: np.ndarray, bbox: Sequence[int]) -> float:
    """
    Compute mean probability for a specific class within a bounding box

    Args:
        prob_channel (np.ndarray): Probability map for the target class
        bbox (Sequence[int]): Spatial coordinates in xywh format
    Returns:
        float: Mean probability value within the specified region
    """
    x, y, w, h = (int(v) for v in bbox)
    H, W = prob_channel.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + max(1, w)), min(H, y + max(1, h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(prob_channel[y0:y1, x0:x1].mean())

def logprobs_to_confidence(token_logprobs: Sequence[float], reduce: str = "mean") -> float:
    """
    Aggregate per-token log-probabilities into a unified confidence score

    Args:
        token_logprobs (Sequence[float]): Natural log probabilities for chosen tokens
        reduce (str): Aggregation method, either mean or min
    Returns:
        float: Unified confidence score in the range zero to one
    """
    lps = [lp for lp in token_logprobs if lp is not None and not math.isinf(lp)]
    if not lps:
        return 0.0
    if reduce == "min":
        return float(math.exp(min(lps)))
    return float(math.exp(sum(lps) / len(lps)))

def group_tokens_to_words(token_strings: Sequence[str]) -> List[List[int]]:
    """
    Group subword tokens into word units based on boundary markers
    """
    groups: List[List[int]] = []
    cur: List[int] = []
    for i, tok in enumerate(token_strings):
        if cur and tok.startswith(WORD_SPACE_MARKERS):
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
    if cur:
        groups.append(cur)
    return groups

def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to raw logits for post-hoc calibration

    Args:
        logits (np.ndarray): Raw class logits before softmax
        temperature (float): Positive scalar factor for scaling
    Returns:
        np.ndarray: Calibrated probabilities after scaling and softmax
    """
    z = logits.astype(np.float64) / max(1e-3, float(temperature))
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

def calibrated_label(probs: Sequence[float], classes: Sequence[str],
                     tau: float = DEFAULT_UNKNOWN_TAU) -> Tuple[str, float, bool]:
    """
    Perform selective classification with a rejection threshold

    Args:
        probs (Sequence[float]): Calibrated class probabilities
        classes (Sequence[str]): Identifier strings for each class
        tau (float): Confidence threshold for the reject option
    Returns:
        Tuple[str, float, bool]: Final label, confidence score, and review flag
    """
    p = np.asarray(probs, dtype=np.float32)
    idx = int(p.argmax())
    conf = float(p[idx])
    if conf < tau:
        return LABEL_UNKNOWN, conf, True
    return classes[idx], conf, False
