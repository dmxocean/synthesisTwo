# -*- coding: utf-8 -*-
"""
Segmentation evaluation executable workflow

This script performs evaluation of the 3-class segmentation models (noise, handwriting, printed) on the held-out test split. It computes IoU, Dice, Precision, Recall, and calibration metrics per historical epoch. Results are written to a JSON file for further analysis and visualization
"""

import os
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

from src.core.config import PATH_DIR_OUTPUT, PATH_FILE_SPLIT, get_artifact_dir, SEG_THRESHOLDS
from src.core.gpu import DeviceManager
from src.core import metrics as M
from src.core import splits as S
from src.preprocessing.normalize import normalise_background
from src.segmentation.inference import load_segmenter, predict_layers
from src.segmentation.data import KEY_LAYERS, IDX_NS, IDX_HW, IDX_PR

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THRESHOLD = SEG_THRESHOLDS[0]


def _empty_counts():
    """
    Return a dictionary of empty binary classification counters
    """
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "inter": 0, "union": 0}


def _read_gt(label_path, shape_hw):
    """
    Read ground truth layers from a multi-frame TIFF file
    """
    if not os.path.exists(label_path):
        return np.zeros((3,) + shape_hw, dtype=bool)
    frames = []
    with Image.open(label_path) as lbl:
        for i in (IDX_NS, IDX_HW, IDX_PR):
            lbl.seek(i)
            frames.append(np.array(lbl) > 0)
    return np.stack(frames, axis=0)


def _compute_block(counts, samples):
    """
    Compute aggregate metrics from counts and subsampled probabilities
    """
    iou, dice, prec, rec, pr, cal = {}, {}, {}, {}, {}, {}
    for ci, key in enumerate(KEY_LAYERS):
        c = counts[ci]
        iou[key] = M.iou_from_counts(c)
        dice[key] = M.dice_from_counts(c)
        p, r, _ = M.prf_from_counts(c)
        prec[key], rec[key] = p, r
        probs = np.asarray(samples[ci]["prob"], dtype=np.float32)
        targets = np.asarray(samples[ci]["target"], dtype=bool)
        if probs.size:
            curve = M.pr_curve(probs, targets)
            curve["ap"] = M.average_precision_pr(curve["precision"], curve["recall"])
            pr[key] = curve
            relib = M.reliability_bins(probs, targets, n_bins=10)
            relib["ece"] = M.expected_calibration_error(probs, targets, n_bins=15)
            cal[key] = relib
    iou["mean"] = float(np.mean([iou[k] for k in KEY_LAYERS]))
    dice["mean"] = float(np.mean([dice[k] for k in KEY_LAYERS]))
    return {
        "iou": iou, "dice": dice, "precision": prec, "recall": rec,
        "pr_curve": pr, "calibration": cal
    }


def main(args):
    """
    Execution logic for the segmentation evaluation pipeline
    """
    device = DeviceManager.get_device()
    DeviceManager.print_hardware_summary()
    model = load_segmenter(args.model, device)

    all_imgs = sorted(glob.glob(os.path.join(args.data, "images", "*_input.png")))
    manifest = S.ensure_split(os.path.join(args.data, "images"), PATH_FILE_SPLIT)
    test_stems = S.stems_for(manifest, "test")
    test_imgs = [p for p in all_imgs if S.page_stem(p) in test_stems]
    if args.limit:
        test_imgs = test_imgs[: args.limit]
    print(f"Test pages: {len(test_imgs)} / {len(all_imgs)} (held-out split)", flush=True)

    # Accumulators for counts and subsamples per epoch and class
    counts = defaultdict(lambda: [_empty_counts() for _ in KEY_LAYERS])
    samples = defaultdict(lambda: [{"prob": [], "target": []} for _ in KEY_LAYERS])
    per_tile_quota = max(1, args.max_pixels // max(1, len(test_imgs)))
    rng = np.random.default_rng(0)

    for n, img_path in enumerate(test_imgs):
        epoch = S.epoch_of(img_path)
        img = np.array(Image.open(img_path).convert("RGB"))
        norm, _ = normalise_background(img)
        probs = predict_layers(model, norm, device)                      # (3, H, W)
        gt = _read_gt(os.path.join(args.data, "labels",
                      os.path.basename(img_path).replace("_input.png", "_label.tiff")),
                      probs.shape[1:])

        flat_idx = None
        for ci in range(len(KEY_LAYERS)):
            pred = probs[ci] > THRESHOLD
            cnt = M.binary_counts(pred, gt[ci])
            for tgt_dict in (counts[epoch][ci], counts["__all__"][ci]):
                for k in tgt_dict:
                    tgt_dict[k] += cnt[k]
            # Subsample pixels shared index across classes for this tile
            if flat_idx is None:
                npx = probs[ci].size
                flat_idx = rng.choice(npx, size=min(per_tile_quota, npx), replace=False)
            pflat = probs[ci].ravel()[flat_idx]
            tflat = gt[ci].ravel()[flat_idx]
            for bucket in (samples[epoch][ci], samples["__all__"][ci]):
                bucket["prob"].append(pflat)
                bucket["target"].append(tflat)
        if (n + 1) % 25 == 0:
            print(f"Progress: {n+1}/{len(test_imgs)}", flush=True)

    # Concat subsamples across pages
    for ep in list(samples.keys()):
        for ci in range(len(KEY_LAYERS)):
            samples[ep][ci]["prob"] = (np.concatenate(samples[ep][ci]["prob"])
                                       if samples[ep][ci]["prob"] else np.array([]))
            samples[ep][ci]["target"] = (np.concatenate(samples[ep][ci]["target"])
                                         if samples[ep][ci]["target"] else np.array([]))

    out = {
        "stage": "segmentation",
        "model": args.model,
        "classes": list(KEY_LAYERS),
        "n_pages": len(test_imgs),
        "threshold": THRESHOLD,
        "overall": _compute_block(counts["__all__"], samples["__all__"]),
        "per_epoch": {ep: _compute_block(counts[ep], samples[ep])
                      for ep in S.EPOCHS if ep in counts},
    }
    out_dir = get_artifact_dir("segmentation", args.model, "metrics")
    out_path = os.path.join(out_dir, "segmentation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote metrics to {out_path}", flush=True)  # Log output path
    print(f"mIoU={out['overall']['iou']['mean']:.4f} "
          + " ".join(f"{k}={out['overall']['iou'][k]:.3f}" for k in KEY_LAYERS), flush=True)  # Log summary metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate the 3-class segmenter on the frozen synthetic test split")
    p.add_argument("--model", default="unet", choices=["segformer", "unet"])
    p.add_argument("--data", default=PATH_DIR_OUTPUT, help="synthetic factory root (images/ + labels/)")
    p.add_argument("--limit", type=int, default=0, help="cap pages for a smoke run")
    p.add_argument("--max-pixels", type=int, default=300_000, help="subsampled pixels per class for PR/calibration")
    main(p.parse_args())
