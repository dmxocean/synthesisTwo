# -*- coding: utf-8 -*-
"""
Prediction pipeline for real archival document clusters

This module performs inference on sampled document images from the clustering stage. It executes segmentation, region extraction, and noise classification across the full sample set, while performing VLM transcription on a subset. The script generates visual overlays and structured transcription JSONs, storing them in standardized output directories without modifying the original data manifest
"""

import os
import json
import random
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.gpu import DeviceManager
from src.core import splits as S
from src.preprocessing.normalize import normalise_background
from src.segmentation.inference import load_segmenter, predict_layers, save_layer_artifacts
from src.segmentation.data import IDX_NS, IDX_HW, IDX_PR
from src.detection.region.extract import extract_regions
from src.detection.noise.predict import load_model as load_noise_model, predict_noise_instances
from src.detection.vlm.qwen import QwenVLM

PATH_DIR_LAYOUTS_SAMPLES = os.path.join(BASE_PATH, "data", "interim", "layouts", "samples")
MANIFEST = os.path.join(PATH_DIR_LAYOUTS_SAMPLES, "manifest.json")
LAYER_COLOR = {"pr": (50, 200, 50), "hw": (235, 60, 60)}  # Green for printed, Red for handwritten
NOISE_COLOR = (0, 160, 255)  # Blue for noise artifacts

def get_artifact_dir(phase: str, model_name: str, artifact_type: str) -> str:
    """
    Retrieve the standardized directory path for model artifacts
    """
    path = os.path.join(BASE_PATH, "outputs", phase, model_name, artifact_type)
    os.makedirs(path, exist_ok=True)
    return path

def _select(per_epoch, vlm_sample, seed, limit):
    """
    Sample document pages from the manifest based on epoch and sample size constraints
    """
    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    by_epoch = defaultdict(list)
    for e in manifest:
        by_epoch[e["epoch"]].append(e)
    
    rng = random.Random(seed)
    selected, vlm_selected = {}, {}
    for ep in S.EPOCHS:
        entries = by_epoch.get(ep, [])
        rng.shuffle(entries)
        chosen = entries[:per_epoch]
        if limit:
            chosen = chosen[:limit]
        selected[ep] = chosen
        vlm_selected[ep] = set(e["file"] for e in chosen[:vlm_sample])
    return selected, vlm_selected

def _draw_boxes(page_np, boxes_with_labels):
    """
    Render bounding boxes and labels onto a document image
    """
    img = Image.fromarray(page_np).convert("RGB")
    d = ImageDraw.Draw(img)
    for (x, y, w, h), color, label in boxes_with_labels:
        d.rectangle([x, y, x + w, y + h], outline=color, width=3)
        if label:
            d.text((x + 2, max(0, y - 12)), label, fill=color)
    return img

def _stem(rel_file):
    """
    Extract the filename stem from a relative path
    """
    return os.path.splitext(os.path.basename(rel_file))[0]

def main(args):
    """
    Execute the cluster prediction pipeline
    """
    device = DeviceManager.get_device()
    DeviceManager.print_hardware_summary()

    selected, vlm_selected = _select(args.per_epoch, args.vlm_sample, args.seed, args.limit)
    n_total = sum(len(v) for v in selected.values())
    n_vlm = sum(len(v) for v in vlm_selected.values())
    print(f"[*] Selected {n_total} pages ({n_vlm} for VLM) across {len(S.EPOCHS)} epochs")

    seg = load_segmenter(args.model, device)
    noise_model = load_noise_model(model_name="noise_resnet", device=device)[0]
    vlm = None if args.no_vlm else QwenVLM(device_map="auto")

    dir_seg = get_artifact_dir("segmentation", args.model, "predictions")
    dir_reg = get_artifact_dir("detection", "regions_classifier", "predictions")
    dir_ns = get_artifact_dir("detection", "noise_resnet", "predictions")
    dir_vlm = get_artifact_dir("detection", "vlm_qwen", "predictions")

    for ep in S.EPOCHS:
        for entry in selected.get(ep, []):
            rel = entry["file"]
            src = os.path.join(PATH_DIR_LAYOUTS_SAMPLES, rel)
            if not os.path.exists(src):
                print(f"[!] missing {src}")
                continue
            
            stem = _stem(rel)
            page = np.array(Image.open(src).convert("RGB"))
            norm, ink = normalise_background(page)
            probs = predict_layers(seg, norm, device)

            save_layer_artifacts(probs, page, ink, stem, os.path.join(dir_seg, ep))  # Export segmentation layers

            pr_regions = extract_regions(probs[IDX_PR], prefix="pr")
            hw_regions = extract_regions(probs[IDX_HW], prefix="hw")
            region_boxes = [(r.bbox, LAYER_COLOR["pr"], r.region_id) for r in pr_regions] + \
                           [(r.bbox, LAYER_COLOR["hw"], r.region_id) for r in hw_regions]
            _save(_draw_boxes(page, region_boxes), os.path.join(dir_reg, ep), f"{stem}_regions.png")

            noise = predict_noise_instances(noise_model, device, page, probs[IDX_NS])
            _save(_draw_boxes(page, [(ni.bbox, NOISE_COLOR, ni.mark_type) for ni in noise]),
                  os.path.join(dir_ns, ep), f"{stem}_noise.png")

            if vlm is not None and rel in vlm_selected[ep]:
                _write_transcriptions(vlm, page, pr_regions, hw_regions, ep, stem, dir_vlm)
            
            print(f"[*] {ep}/{stem}: pr={len(pr_regions)} hw={len(hw_regions)} noise={len(noise)}")

    sel_path = os.path.join(get_artifact_dir("indexing", "clustering", "logs"), "selection.json")
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed, "per_epoch": args.per_epoch, "vlm_sample": args.vlm_sample,
            "model": args.model, "n_total": n_total, "n_vlm": n_vlm,
            "selected": {ep: [e["file"] for e in selected[ep]] for ep in selected},
            "vlm_selected": {ep: sorted(vlm_selected[ep]) for ep in vlm_selected},
        }, f, indent=2, ensure_ascii=False)
    print(f"[*] Wrote selection to {sel_path}")

def _save(img, out_dir, name):
    """
    Save an image object to the specified directory
    """
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, name))

def _write_transcriptions(vlm, page_np, pr_regions, hw_regions, epoch, stem, base_dir):
    """
    Generate and persist region-level transcriptions using the VLM
    """
    records = []
    for layer, regions in (("pr", pr_regions), ("hw", hw_regions)):
        for order, r in enumerate(regions):
            x, y, w, h = r.bbox
            crop = Image.fromarray(page_np[y:y + h, x:x + w])
            res = vlm.transcribe_region(crop)
            records.append({
                "region_id": r.region_id, "layer": layer, "reading_order": order,
                "bbox": r.bbox, "text": res[0], "confidence": float(res[1]),
            })
    
    out_dir = os.path.join(base_dir, epoch)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump({"page": stem, "epoch": epoch, "regions": records}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Predict on sampled real cluster pages per epoch")
    p.add_argument("--model", default="unet", choices=["segformer", "unet"], help="Backbone model")
    p.add_argument("--per-epoch", type=int, default=100, help="Pages per epoch")
    p.add_argument("--vlm-sample", type=int, default=25, help="VLM samples per epoch")
    p.add_argument("--no-vlm", action="store_true", help="Disable transcription")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--limit", type=int, default=0, help="Per-epoch limit")
    args = p.parse_args()
    main(args)
