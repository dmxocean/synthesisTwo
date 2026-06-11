# -*- coding: utf-8 -*-
"""
Shared evaluation metrics for archival records processing

This module provides a collection of pure-Python and NumPy-based metrics for segmentation, detection, and classification. It handles confusion matrices, average precision, calibration error, and transcription accuracy without external metric libraries
"""

import os
import math
import unicodedata
from typing import Dict, List, Sequence, Tuple

import numpy as np

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORAGE_DIR = os.path.join(BASE_PATH, "storage")
DATA_DIR = os.path.join(BASE_PATH, "data")

EPS = 1e-9
COCO_IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]

def binary_counts(pred_bool: np.ndarray, target_bool: np.ndarray) -> Dict[str, int]:
    """
    Confusion counts and intersection/union for one binary mask pair

    The function accumulates counts across images to maintain consistency in streaming calculations
    """
    p = np.asarray(pred_bool).astype(bool)
    t = np.asarray(target_bool).astype(bool)
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "inter": tp, "union": tp + fp + fn}

def iou_from_counts(counts: Dict[str, int]) -> float:
    """
    Calculate IoU from accumulated binary counts
    """
    return float(counts["inter"] / (counts["union"] + EPS))

def dice_from_counts(counts: Dict[str, int]) -> float:
    """
    Calculate Dice coefficient from accumulated binary counts
    """
    inter = counts["inter"]
    return float(2 * inter / (2 * inter + counts["fp"] + counts["fn"] + EPS))

def prf_from_counts(counts: Dict[str, int]) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 from binary counts
    """
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    return float(precision), float(recall), float(f1)

def pr_curve(probs: np.ndarray, target_bool: np.ndarray,
             thresholds: Sequence[float] | None = None) -> Dict[str, List[float]]:
    """
    Precision and recall at a sweep of probability thresholds

    Args:
        probs (np.ndarray): Predicted probabilities in range zero to one
        target_bool (np.ndarray): Ground-truth binary mask
        thresholds (Sequence[float]): Thresholds to evaluate during the sweep
    Returns:
        Dict[str, List[float]]: Parallel lists of thresholds, precision, and recall
    """
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(1, 20)]
    p = np.asarray(probs).ravel().astype(np.float32)
    t = np.asarray(target_bool).ravel().astype(bool)
    n_pos = int(t.sum())
    precision, recall = [], []
    for thr in thresholds:
        pred = p >= thr
        tp = int(np.logical_and(pred, t).sum())
        fp = int(np.logical_and(pred, ~t).sum())
        precision.append(float(tp / (tp + fp + EPS)))
        recall.append(float(tp / (n_pos + EPS)))
    return {"thresholds": list(map(float, thresholds)), "precision": precision, "recall": recall}

def average_precision_pr(precision: Sequence[float], recall: Sequence[float]) -> float:
    """
    Compute area under the precision-recall curve using the trapezoidal rule
    """
    r = np.asarray(recall, dtype=np.float64)
    p = np.asarray(precision, dtype=np.float64)
    order = np.argsort(r)
    r, p = r[order], p[order]
    return float(np.trapz(p, r))

def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Calculate intersection over union for two boxes in xywh format
    """
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return float(inter / (union + EPS))

def match_detections(pred_boxes: Sequence[Sequence[float]], scores: Sequence[float],
                     gt_boxes: Sequence[Sequence[float]], iou_thr: float
                     ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform greedy score-ordered matching of predicted boxes to ground-truth

    Args:
        pred_boxes (Sequence[Sequence[float]]): Predicted boxes in xywh format
        scores (Sequence[float]): Confidence scores for predicted boxes
        gt_boxes (Sequence[Sequence[float]]): Ground-truth boxes in xywh format
        iou_thr (float): IoU threshold for matching a prediction to ground-truth
    Returns:
        Tuple[np.ndarray, np.ndarray, int]: Binary arrays for tp/fp and total ground-truth count
    """
    n_pred = len(pred_boxes)
    tp = np.zeros(n_pred, dtype=np.float64)
    fp = np.zeros(n_pred, dtype=np.float64)
    order = np.argsort(-np.asarray(scores, dtype=np.float64)) if n_pred else np.array([], dtype=int)
    matched = set()
    for rank, idx in enumerate(order):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched:
                continue
            iou = box_iou(pred_boxes[idx], gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp[rank] = 1.0
            matched.add(best_j)
        else:
            fp[rank] = 1.0
    return tp, fp, len(gt_boxes)

def average_precision(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    """
    Compute average precision using the VOC2010 all-point integration method
    """
    if n_gt == 0:
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / (n_gt + EPS)
    precision = cum_tp / (cum_tp + cum_fp + EPS)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def ap_at_iou(pred_boxes: Sequence[Sequence[float]], scores: Sequence[float],
              gt_boxes: Sequence[Sequence[float]], iou_thr: float) -> float:
    """
    Convenience function for calculating AP at a specific IoU threshold

    Args:
        pred_boxes (Sequence[Sequence[float]]): Predicted boxes in xywh format
        scores (Sequence[float]): Confidence scores for predicted boxes
        gt_boxes (Sequence[Sequence[float]]): Ground-truth boxes in xywh format
        iou_thr (float): IoU threshold for matching
    Returns:
        float: Computed average precision
    """
    tp, fp, n_gt = match_detections(pred_boxes, scores, gt_boxes, iou_thr)
    return average_precision(tp, fp, n_gt)

def coco_map(preds_by_class: Dict[str, Tuple[list, list]],
             gts_by_class: Dict[str, list]) -> Dict[str, float]:
    """
    Calculate COCO-style mAP averaged over classes and IoU thresholds

    Args:
        preds_by_class (Dict[str, Tuple[list, list]]): Mapping of class to box-score pairs
        gts_by_class (Dict[str, list]): Mapping of class to ground-truth boxes
    Returns:
        Dict[str, float]: Results for AP50, AP75, and overall mAP
    """
    classes = sorted(set(gts_by_class) | set(preds_by_class))
    def _mean_ap(iou):
        per = []
        for c in classes:
            boxes, scores = preds_by_class.get(c, ([], []))
            per.append(ap_at_iou(boxes, scores, gts_by_class.get(c, []), iou))
        return float(np.mean(per)) if per else 0.0
    ap50 = _mean_ap(0.5)
    ap75 = _mean_ap(0.75)
    map_ = float(np.mean([_mean_ap(t) for t in COCO_IOU_THRESHOLDS]))
    return {"ap50": ap50, "ap75": ap75, "map": map_}

def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> List[List[int]]:
    """
    Generate a confusion matrix as a nested list for JSON serialization

    Args:
        y_true (Sequence[int]): Ground-truth class indices
        y_pred (Sequence[int]): Predicted class indices
        n_classes (int): Total number of unique classes
    Returns:
        List[List[int]]: Matrix where rows represent true classes and columns represent predicted
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm.tolist()

def prf_from_confusion(cm: Sequence[Sequence[int]]) -> Dict[str, object]:
    """
    Calculate precision, recall, and F1 from a confusion matrix
    """
    m = np.asarray(cm, dtype=np.float64)
    tp = np.diag(m)
    fp = m.sum(axis=0) - tp
    fn = m.sum(axis=1) - tp
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "accuracy": float(tp.sum() / (m.sum() + EPS)),
    }

def reliability_bins(confidences: Sequence[float], correct: Sequence[bool],
                     n_bins: int = 10) -> Dict[str, List[float]]:
    """
    Partition confidence scores into equal-width bins for reliability analysis

    Args:
        confidences (Sequence[float]): Confidence scores for predictions
        correct (Sequence[bool]): Ground-truth correctness for predictions
        n_bins (int): Number of bins to create
    Returns:
        Dict[str, List[float]]: Accuracy and confidence metrics per bin
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, bin_conf, bin_acc, bin_count = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        n = int(sel.sum())
        centers.append(float((lo + hi) / 2))
        bin_count.append(n)
        bin_conf.append(float(conf[sel].mean()) if n else 0.0)
        bin_acc.append(float(corr[sel].mean()) if n else 0.0)
    return {"bin_center": centers, "bin_confidence": bin_conf,
            "bin_accuracy": bin_acc, "bin_count": bin_count}

def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                               n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error from binned confidence scores
    """
    bins = reliability_bins(confidences, correct, n_bins)
    total = sum(bins["bin_count"]) or 1
    ece = 0.0
    for n, acc, cf in zip(bins["bin_count"], bins["bin_accuracy"], bins["bin_confidence"]):
        ece += (n / total) * abs(acc - cf)
    return float(ece)

def _softmax(z: np.ndarray) -> np.ndarray:
    """
    Compute the softmax function for a vector of logits
    """
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + EPS)

def fit_temperature(logits: np.ndarray, labels: Sequence[int],
                    grid: Sequence[float] | None = None) -> float:
    """
    Find the optimal temperature scaling factor to minimize NLL

    Args:
        logits (np.ndarray): Raw class logits from the model
        labels (Sequence[int]): Ground-truth class indices
        grid (Sequence[float]): Candidate temperatures for search
    Returns:
        float: The temperature value that maximizes calibration
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    rows = np.arange(len(y))

    def nll(t_val):
        p = _softmax(z / t_val)
        return float(-np.log(p[rows, y] + EPS).mean())

    candidates = list(grid) if grid is not None else list(np.linspace(0.5, 5.0, 19))
    best_t = min(candidates, key=nll)
    fine = np.linspace(max(0.05, best_t - 0.25), best_t + 0.25, 21)
    best_t = min(list(fine), key=nll)
    return float(best_t)

def risk_coverage(confidences: Sequence[float], correct: Sequence[bool]) -> Dict[str, List[float]]:
    """
    Calculate the risk-coverage curve for a selective prediction strategy
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-conf)
    corr_sorted = corr[order]
    conf_sorted = conf[order]
    n = len(conf)
    if n == 0:
        return {"coverage": [], "risk": [], "accuracy": [], "threshold": []}
    cum_correct = np.cumsum(corr_sorted)
    k = np.arange(1, n + 1)
    accuracy = cum_correct / k
    return {
        "coverage": (k / n).tolist(),
        "risk": (1.0 - accuracy).tolist(),
        "accuracy": accuracy.tolist(),
        "threshold": conf_sorted.tolist(),
    }

def _edit_distance(a: Sequence, b: Sequence) -> int:
    """
    Compute Levenshtein distance between two sequences of tokens
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]

def normalize_text(s: str) -> str:
    """
    Apply standard normalization to transcription text strings
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s.lower())
    return " ".join(s.split())

def cer(reference: str, hypothesis: str, normalized: bool = False) -> float:
    """
    Calculate Character Error Rate for a pair of strings
    """
    if normalized:
        reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    ref = list(reference)
    return float(_edit_distance(ref, list(hypothesis)) / max(1, len(ref)))

def wer(reference: str, hypothesis: str, normalized: bool = False) -> float:
    """
    Calculate Word Error Rate for a pair of strings
    """
    if normalized:
        reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    ref = reference.split()
    return float(_edit_distance(ref, hypothesis.split()) / max(1, len(ref)))

def hit_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """
    Determine if any relevant identifier appears in the top-k results
    """
    rel = set(relevant)
    return 1.0 if any(r in rel for r in list(retrieved)[:k]) else 0.0

def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """
    Calculate the fraction of relevant identifiers found in the top-k results
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    top = set(list(retrieved)[:k])
    return float(len(rel & top) / len(rel))

def precision_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """
    Calculate the precision of results within the top-k retrieved items
    """
    retrieved_k = retrieved[:k]
    if k <= 0:
        return 0.0
    if not retrieved_k:
        return 0.0
    expected_set = set(expected)
    hits = sum(1 for rid in retrieved_k if rid in expected_set)
    return hits / float(k)

def mean_reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """
    Compute the mean reciprocal rank for the first relevant result
    """
    rel = set(relevant)
    for rank, rid in enumerate(retrieved, start=1):
        if rid in rel:
            return float(1.0 / rank)
    return 0.0

def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """
    Compute normalized discounted cumulative gain at rank k
    """
    rel = set(relevant)
    dcg = 0.0
    for i, rid in enumerate(list(retrieved)[:k]):
        if rid in rel:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else 0.0
