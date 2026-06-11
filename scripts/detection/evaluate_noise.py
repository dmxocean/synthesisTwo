# -*- coding: utf-8 -*-
"""
Evaluation pipeline for the noise classification model

This module assesses the performance of the ResNet-18 noise classifier on the held-out test split. It calculates per-class precision, recall, and F1 scores, generates confusion matrices, and performs temperature calibration to evaluate model confidence and risk-coverage characteristics
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.gpu import DeviceManager
from src.core import metrics as M
from src.core import splits as S
from src.core.confidence import DEFAULT_UNKNOWN_TAU
from src.detection.noise.predict import load_model
from src.detection.noise.classifier import CLASSES_NOISE, SIZE_INPUT

PATH_DIR_OUTPUT = os.path.join(BASE_PATH, "data", "synthetic", "factory")
PATH_FILE_SPLIT = os.path.join(PATH_DIR_OUTPUT, "split.json")

MAP_CAT_ID = {3: "circles", 4: "lines", 5: "crosses", 6: "marks", 7: "stamps"}  # Map COCO category IDs to noise types
LABEL_OF = {name: CLASSES_NOISE.index(name) for name in CLASSES_NOISE}  # Create label index mapping
_NORM = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Standard ImageNet normalization

def get_artifact_dir(phase: str, model_name: str, artifact_type: str) -> str:
    """
    Retrieve the standardized directory path for model artifacts

    Args:
        phase (str): Pipeline stage such as segmentation or detection
        model_name (str): Identifier for the specific model architecture
        artifact_type (str): Category of artifact like weights or metrics
    Returns:
        str: Absolute path to the resolved and created directory
    """
    path = os.path.join(BASE_PATH, "outputs", phase, model_name, artifact_type)
    os.makedirs(path, exist_ok=True)
    return path

def _crop_tensor(page_pil, x, y, w, h):
    """
    Transform image crop into a normalized tensor
    """
    crop = page_pil.crop((x, y, x + w, y + h)).resize((SIZE_INPUT, SIZE_INPUT), Image.BILINEAR)
    return _NORM(TF.to_tensor(crop))

@torch.no_grad()
def _logits_for_image(model, device, page_pil, boxes):
    """
    Compute model logits for a collection of image crops
    """
    batch = torch.stack([_crop_tensor(page_pil, *b) for b in boxes]).to(device)
    return model(batch).float().cpu().numpy()

def _calibration_block(logits, labels):
    """
    Fit temperature on calibration split and report reliability metrics

    Args:
        logits (np.ndarray): raw model outputs
        labels (np.ndarray): ground truth class indices
    Returns:
        dict: calibration results including temperature and ECE
    """
    n = len(labels)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    cal, evl = idx[: n // 2], idx[n // 2:]  # Split into calibration and evaluation sets
    T = M.fit_temperature(logits[cal], labels[cal]) if len(cal) else 1.0

    def _bins(z, lab, temp):
        p = M._softmax(z / temp)
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == lab)
        rb = M.reliability_bins(conf, correct, n_bins=10)
        rb["ece"] = M.expected_calibration_error(conf, correct, n_bins=15)
        return rb

    return {
        "temperature": float(T),
        "before": _bins(logits[evl], labels[evl], 1.0) if len(evl) else {},
        "after":  _bins(logits[evl], labels[evl], T) if len(evl) else {},
    }

def main(args):
    """
    Execute the noise classifier evaluation pipeline

    Args:
        args (argparse.Namespace): command line arguments containing model and data paths
    Returns:
        None
    """
    device = DeviceManager.get_device()
    DeviceManager.print_hardware_summary()
    model, device = load_model(model_name=args.model, device=device)

    with open(os.path.join(args.data, "coco_instances.json"), "r", encoding="utf-8") as f:
        coco = json.load(f)  # Load COCO annotations for the test set

    file_of = {im["id"]: im["file_name"] for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] in MAP_CAT_ID:
            anns_by_img[a["image_id"]].append(a)

    test_stems = S.stems_for(S.ensure_split(os.path.join(args.data, "images"), PATH_FILE_SPLIT), "test")

    y_true, y_pred, conf_all = [], [], []
    logits_all, labels_all = [], []
    ep_true, ep_pred = defaultdict(list), defaultdict(list)
    n_img = 0
    
    for img_id, anns in anns_by_img.items():
        fname = file_of.get(img_id)
        if not fname or S.page_stem(fname) not in test_stems:
            continue
        img_path = os.path.join(args.data, "images", fname)
        if not os.path.exists(img_path):
            continue
        epoch = S.epoch_of(fname)
        page = Image.open(img_path).convert("RGB")
        boxes = [[int(v) for v in a["bbox"]] for a in anns]
        labels = [LABEL_OF[MAP_CAT_ID[a["category_id"]]] for a in anns]
        logits = _logits_for_image(model, device, page, boxes)
        probs = M._softmax(logits)
        preds = probs.argmax(axis=1)
        
        for lab, pr, p in zip(labels, preds, probs):
            y_true.append(lab)
            y_pred.append(int(pr))
            conf_all.append(float(p.max()))
            ep_true[epoch].append(lab)
            ep_pred[epoch].append(int(pr))
        
        logits_all.append(logits)
        labels_all.extend(labels)
        n_img += 1
        if args.limit and n_img >= args.limit:
            break

    logits_all = np.concatenate(logits_all) if logits_all else np.zeros((0, len(CLASSES_NOISE)))
    labels_all = np.asarray(labels_all)
    correct = [int(t == p) for t, p in zip(y_true, y_pred)]

    cm = M.confusion_matrix(y_true, y_pred, len(CLASSES_NOISE))
    out = {
        "stage": "noise",
        "model": args.model,
        "classes": CLASSES_NOISE,
        "n_crops": len(y_true), "n_pages": n_img,
        "tau": DEFAULT_UNKNOWN_TAU,
        "overall": {
            "confusion_matrix": cm,
            **M.prf_from_confusion(cm),
            "calibration": _calibration_block(logits_all, labels_all) if len(labels_all) else {},
            "risk_coverage": M.risk_coverage(conf_all, correct),
        },
        "per_epoch": {},
    }
    for ep in S.EPOCHS:
        if ep in ep_true:
            cm_e = M.confusion_matrix(ep_true[ep], ep_pred[ep], len(CLASSES_NOISE))
            out["per_epoch"][ep] = {"confusion_matrix": cm_e, **M.prf_from_confusion(cm_e)}

    out_dir = get_artifact_dir("detection", args.model, "metrics")
    out_path = os.path.join(out_dir, "noise.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    print(f"[*] Wrote {out_path}", flush=True)
    print(f"[*] macro_f1={out['overall']['macro_f1']:.3f} accuracy={out['overall']['accuracy']:.3f} "
          f"T={out['overall']['calibration'].get('temperature', 1.0):.2f}", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate the ResNet-18 noise classifier on the frozen test split")
    p.add_argument("--model", default="noise_resnet", help="Model name for paths (default: noise_resnet)")
    p.add_argument("--data", default=PATH_DIR_OUTPUT)
    p.add_argument("--limit", type=int, default=0, help="cap pages for a smoke run")
    main(p.parse_args())
