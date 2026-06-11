# -*- coding: utf-8 -*-
"""
Evaluation pipeline for region extraction performance

This module assesses the quality of document region extraction by calculating coverage and purity metrics alongside standard detection benchmarks like AP50 and mAP. It evaluates the ability of the extraction logic to capture text pixels without bleed and provides diagnostic histograms for over and under-segmentation across different historical epochs
"""

import os
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.gpu import DeviceManager
from src.core import metrics as M
from src.core import splits as S
from src.core.confidence import region_confidence
from src.preprocessing.normalize import normalise_background
from src.segmentation.inference import load_segmenter, predict_layers
from src.segmentation.data import IDX_HW, IDX_PR
from src.detection.region.extract import extract_regions

PATH_DIR_OUTPUT = os.path.join(BASE_PATH, "data", "synthetic", "factory")
PATH_FILE_SPLIT = os.path.join(PATH_DIR_OUTPUT, "split.json")

LAYER_OF = {"printed": ("pr", IDX_PR), "handwritten": ("hw", IDX_HW)}  # Map annotation categories to layer indices
IOU_LINK = 0.10  # Overlap threshold for fragmentation diagnostics

def get_artifact_dir(phase: str, model_name: str, artifact_type: str) -> str:
    """
    Retrieve the standardized directory path for model artifacts
    """
    path = os.path.join(BASE_PATH, "outputs", phase, model_name, artifact_type)
    os.makedirs(path, exist_ok=True)
    return path

def _gt_boxes(ann_path):
    """
    Load ground truth bounding boxes from annotation files
    """
    out = {"pr": [], "hw": []}
    if not os.path.exists(ann_path):
        return out
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for a in data.get("annotations", []):
            if a["category"] in LAYER_OF:
                out[LAYER_OF[a["category"]][0]].append([int(v) for v in a["bbox"]])
    return out

def _read_gt_mask(label_path, shape_hw):
    """
    Extract binary masks for printed and handwritten layers from TIFF files
    """
    out = {"hw": np.zeros(shape_hw, bool), "pr": np.zeros(shape_hw, bool)}
    if not os.path.exists(label_path):
        return out
    with Image.open(label_path) as lbl:
        for layer, frame in (("hw", IDX_HW), ("pr", IDX_PR)):
            lbl.seek(frame)
            out[layer] = np.array(lbl) > 0
    return out

def _union_mask(boxes, shape_hw):
    """
    Generate a unified binary coverage mask from predicted bounding boxes
    """
    m = np.zeros(shape_hw, bool)
    for x, y, w, h in boxes:
        m[max(0, y):y + h, max(0, x):x + w] = True
    return m

def _seg_histogram(pred_boxes, gt_boxes):
    """
    Calculate fragmentation metrics for predicted and ground truth boxes
    """
    gt_per_pred = [sum(1 for g in gt_boxes if M.box_iou(p, g) > IOU_LINK) for p in pred_boxes]
    pred_per_gt = [sum(1 for p in pred_boxes if M.box_iou(p, g) > IOU_LINK) for g in gt_boxes]
    return gt_per_pred, pred_per_gt

def _finalise(layer_store):
    """
    Compute aggregate metrics from collected per-page results
    """
    block = {}
    preds_by_class, gts_by_class = {}, {}
    for layer, s in layer_store.items():
        pred_boxes = [b for boxes, _ in s["pred"] for b in boxes]
        pred_scores = [sc for _, scores in s["pred"] for sc in scores]
        gt_boxes = s["gt"]
        preds_by_class[layer] = (pred_boxes, pred_scores)
        gts_by_class[layer] = gt_boxes
        tp, fp, n_gt = M.match_detections(pred_boxes, pred_scores, gt_boxes, 0.5)
        n_tp = float(tp.sum())
        precision = n_tp / (len(pred_boxes) + M.EPS)
        recall = n_tp / (n_gt + M.EPS)
        f1 = 2 * precision * recall / (precision + recall + M.EPS)
        cov = s["cov"]
        block[layer] = {
            "coverage": float(cov["inter"] / (cov["gt"] + M.EPS)),
            "purity": float(cov["inter"] / (cov["pred"] + M.EPS)),
            "n_pred": len(pred_boxes),
            "n_gt": n_gt,
            "precision@0.5": float(precision),
            "recall@0.5": float(recall),
            "f1@0.5": float(f1),
            "gt_per_pred_hist": _hist(s["gpp"]),
            "pred_per_gt_hist": _hist(s["ppg"]),
        }
    block["map"] = M.coco_map(preds_by_class, gts_by_class)
    return block

def _hist(values, max_bin=6):
    """
    Create a frequency distribution of integer values
    """
    h = defaultdict(int)
    for v in values:
        h[min(int(v), max_bin)] += 1
    return {str(k): h[k] for k in sorted(h)}

def main(args):
    """
    Execute the region extraction evaluation pipeline
    """
    device = DeviceManager.get_device()
    DeviceManager.print_hardware_summary()
    model = load_segmenter(args.model, device)
    output_domain_model = "regions_classifier"

    all_imgs = sorted(glob.glob(os.path.join(args.data, "images", "*_input.png")))
    test_stems = S.stems_for(S.ensure_split(os.path.join(args.data, "images"), PATH_FILE_SPLIT), "test")
    test_imgs = [p for p in all_imgs if S.page_stem(p) in test_stems]
    if args.limit:
        test_imgs = test_imgs[: args.limit]
    print(f"[*] Test pages: {len(test_imgs)} / {len(all_imgs)}")

    def _new_store():
        return {layer: {"pred": [], "gt": [], "gpp": [], "ppg": [],
                        "cov": {"inter": 0, "gt": 0, "pred": 0}} for layer in ("pr", "hw")}
    stores = defaultdict(_new_store)  # Keyed by epoch and global aggregator

    for n, img_path in enumerate(test_imgs):
        epoch = S.epoch_of(img_path)
        img = np.array(Image.open(img_path).convert("RGB"))
        norm, _ = normalise_background(img)
        probs = predict_layers(model, norm, device)
        ann = os.path.join(args.data, "annotations",
                           os.path.basename(img_path).replace("_input.png", ".json"))
        gt = _gt_boxes(ann)
        gt_mask = _read_gt_mask(os.path.join(args.data, "labels",
                  os.path.basename(img_path).replace("_input.png", "_label.tiff")), probs.shape[1:])

        for layer, ch in (("pr", IDX_PR), ("hw", IDX_HW)):
            regions = extract_regions(probs[ch], prefix=layer)
            boxes = [r.bbox for r in regions]
            scores = [region_confidence(probs[ch], b) for b in boxes]
            gpp, ppg = _seg_histogram(boxes, gt[layer])
            pred_mask = _union_mask(boxes, probs.shape[1:])
            inter = int(np.logical_and(pred_mask, gt_mask[layer]).sum())
            for store in (stores[epoch], stores["__all__"]):
                store[layer]["pred"].append((boxes, scores))
                store[layer]["gt"].extend(gt[layer])
                store[layer]["gpp"].extend(gpp)
                store[layer]["ppg"].extend(ppg)
                store[layer]["cov"]["inter"] += inter
                store[layer]["cov"]["gt"] += int(gt_mask[layer].sum())
                store[layer]["cov"]["pred"] += int(pred_mask.sum())
        if (n + 1) % 25 == 0:
            print(f"[*] {n+1}/{len(test_imgs)}")

    out = {
        "stage": "regions",
        "model": output_domain_model,
        "n_pages": len(test_imgs),
        "layers": ["pr", "hw"],
        "iou_link": IOU_LINK,
        "overall": _finalise(stores["__all__"]),
        "per_epoch": {ep: _finalise(stores[ep]) for ep in S.EPOCHS if ep in stores},
    }
    
    out_dir = get_artifact_dir("detection", output_domain_model, "metrics")
    out_path = os.path.join(out_dir, "regions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    print(f"[*] Wrote {out_path}")
    ov = out["overall"]
    print(f"[*] coverage pr={ov['pr']['coverage']:.3f} hw={ov['hw']['coverage']:.3f} | "
          f"purity pr={ov['pr']['purity']:.3f} hw={ov['hw']['purity']:.3f} | "
          f"AP50={ov['map']['ap50']:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate region extraction as detection on the frozen test split")
    p.add_argument("--model", default="unet", choices=["segformer", "unet"], help="Backbone model for probabilities")
    p.add_argument("--data", default=PATH_DIR_OUTPUT)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
