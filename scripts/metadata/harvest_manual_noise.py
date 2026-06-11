# -*- coding: utf-8 -*-
"""
High-Precision Harvester for Manual Noise Assets

This module uses COCO polygons for surgical cropping and captures advanced geometric metadata for noise instances. It analyzes hollow ratios and orientation to support precise noise synthesis in the document generation pipeline
"""

import os
import json
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.metadata.harvester_core import HarvesterCore, PATH_TRAIN_IMAGES, relpath_posix

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_OUT_ASSETS = os.path.join(BASE_PATH, "data", "assets", "manual", "noise")
PATH_OUT_METADATA = os.path.join(BASE_PATH, "data", "metadata", "manual_noise.json")

CATEGORY_MAP = {
    "Crosses":  "crosses",
    "Lines":    "lines",
    "Marks":    "marks",
    "Crossout": "marks",
    "Stamps":   "stamps",
    "Circles":  "circles"
}


def analyze_hollow_ratio(img_rgba):
    """
    Analyzes the ratio of internal gaps (holes) to the solid ink area

    Perfect for identifying 'E-marks' or hollow stamps
    """
    alpha = np.array(img_rgba)[:, :, 3]  # Isolate alpha channel
    _, binary = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours with hierarchy
    
    if not contours or hierarchy is None:
        return 0.0, False

    solid_area = 0
    hole_area = 0
    has_holes = False

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if hierarchy[0][i][3] == -1:  # External contour
            solid_area += area
        else:  # Internal contour (hole)
            hole_area += area
            has_holes = True

    hollow_ratio = hole_area / solid_area if solid_area > 0 else 0
    return round(float(hollow_ratio), 3), has_holes


def process_noise_task(task):
    """
    Worker function for processing a single noise asset
    """
    subtype, img_path, segmentation, anno_id, img_filename, path_root = task
    
    asset_img = HarvesterCore.surgical_crop(img_path, segmentation)  # Surgical Crop
    if not asset_img:
        return None

    geo = HarvesterCore.analyze_noise_geometry(segmentation)  # Geometric Intelligence
    hollow_ratio, is_hollow = analyze_hollow_ratio(asset_img)
    
    asset_id = f"{subtype}_{anno_id:04d}"  # Save Asset
    save_dir = os.path.join(PATH_OUT_ASSETS, subtype)
    os.makedirs(save_dir, exist_ok=True)
    
    asset_path = os.path.join(save_dir, f"{asset_id}.png")
    asset_img.save(asset_path)

    epoch = HarvesterCore.get_epoch(img_filename)  # Record Metadata
    
    return {
        "asset_id": asset_id,
        "original_file": img_filename,
        "epoch": epoch,
        "path_rel": relpath_posix(asset_path, path_root),
        "category": "noise",
        "subtype": subtype,
        "geometry": {
            "polygon": segmentation,
            "angle": geo["angle"],
            "solidity": geo["solidity"],
            "hollow_ratio": hollow_ratio,
            "is_hollow": is_hollow,
            "aspect_ratio": round(geo["width_oriented"] / geo["height_oriented"], 3) if geo["height_oriented"] > 0 else 1.0,
            "dims": {"w": int(asset_img.width), "h": int(asset_img.height)},
            "oriented_dims": {"w": geo["width_oriented"], "h": geo["height_oriented"]}
        }
    }


def run(workers=4):
    """
    Execute the noise harvest pipeline in parallel
    """
    print("[*] Loading COCO database")
    coco = HarvesterCore.load_coco()
    if not coco:
        return

    image_lookup = {img["id"]: img for img in coco["images"]}  # Build lookup for images
    cat_lookup = {cat["id"]: cat["name"] for cat in coco["categories"]}

    tasks = []
    print("[*] Planning Noise harvest")

    for anno in coco["annotations"]:
        cat_name = cat_lookup.get(anno["category_id"])
        if cat_name not in CATEGORY_MAP:
            continue
        
        subtype = CATEGORY_MAP[cat_name]
        img_info = image_lookup.get(anno["image_id"])
        if not img_info:
            continue

        img_path = os.path.join(PATH_TRAIN_IMAGES, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        
        img_filename = img_info.get("extra", {}).get("name", img_info["file_name"])
        tasks.append((subtype, img_path, anno["segmentation"], anno["id"], img_filename, BASE_PATH))

    metadata = []
    print(f"[*] Harvesting {len(tasks)} Manual Noise assets using {workers} workers")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_noise_task, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Noise Harvest"):
            res = future.result()
            if res:
                metadata.append(res)

    os.makedirs(os.path.dirname(PATH_OUT_METADATA), exist_ok=True)  # Save final metadata
    with open(PATH_OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[*] Noise metadata saved to: {PATH_OUT_METADATA}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Noise Harvester")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    run(workers=args.workers)
