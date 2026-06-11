# -*- coding: utf-8 -*-
"""
Harvester for Manual Handwritten Assets

This module extracts authentic handwriting (HTR) snippets from original scans using COCO polygons. It performs surgical cropping based on segmentation masks and records geometric metadata to support downstream synthesis and training tasks
"""

import os
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.metadata.harvester_core import HarvesterCore, PATH_TRAIN_IMAGES, relpath_posix

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_OUT_ASSETS = os.path.join(BASE_PATH, "data", "assets", "manual", "handwritten")
PATH_OUT_METADATA = os.path.join(BASE_PATH, "data", "metadata", "manual_htr.json")


def process_htr_task(task):
    """
    Worker function for processing a single HTR asset
    """
    img_path, segmentation, anno_id, img_filename, path_root = task
    
    asset_img = HarvesterCore.surgical_crop(img_path, segmentation)  # Surgical Crop
    if not asset_img:
        return None

    asset_id = f"htr_{anno_id:04d}"  # Save Asset
    os.makedirs(PATH_OUT_ASSETS, exist_ok=True)
    
    asset_path = os.path.join(PATH_OUT_ASSETS, f"{asset_id}.png")
    asset_img.save(asset_path)

    epoch = HarvesterCore.get_epoch(img_filename)  # Record Metadata
    
    return {
        "asset_id": asset_id,
        "original_file": img_filename,
        "epoch": epoch,
        "path_rel": relpath_posix(asset_path, path_root),
        "category": "handwritten",
        "polygon": segmentation,
        "dims": {"w": int(asset_img.width), "h": int(asset_img.height)}
    }


def run(workers=4):
    """
    Execute the HTR harvest pipeline in parallel
    """
    print("[*] Loading COCO database")
    coco = HarvesterCore.load_coco()
    if not coco:
        return

    image_lookup = {img["id"]: img for img in coco["images"]}
    cat_lookup = {cat["id"]: cat["name"] for cat in coco["categories"]}

    tasks = []
    print("[*] Planning Handwritten (HTR) harvest")

    for anno in coco["annotations"]:
        cat_name = cat_lookup.get(anno["category_id"])
        if cat_name != "HTR":
            continue
        
        img_info = image_lookup.get(anno["image_id"])
        if not img_info:
            continue

        img_path = os.path.join(PATH_TRAIN_IMAGES, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        
        img_filename = img_info.get("extra", {}).get("name", img_info["file_name"])
        tasks.append((img_path, anno["segmentation"], anno["id"], img_filename, BASE_PATH))

    metadata = []
    print(f"[*] Harvesting {len(tasks)} HTR assets using {workers} workers")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_htr_task, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="HTR Harvest"):
            res = future.result()
            if res:
                metadata.append(res)

    os.makedirs(os.path.dirname(PATH_OUT_METADATA), exist_ok=True)
    with open(PATH_OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[*] Handwritten metadata saved to: {PATH_OUT_METADATA}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel HTR Harvester")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    run(workers=args.workers)
