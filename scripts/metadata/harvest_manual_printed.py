# -*- coding: utf-8 -*-
"""
Harvester for Manual Printed Assets

This module extracts authentic typewriter and printed snippets (OCR) from original scans using COCO polygons. It performs surgical cropping and records geometric metadata to provide high-quality assets for synthetic document generation
"""

import os
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.metadata.harvester_core import HarvesterCore, PATH_TRAIN_IMAGES, relpath_posix

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_OUT_ASSETS = os.path.join(BASE_PATH, "data", "assets", "manual", "printed")
PATH_OUT_METADATA = os.path.join(BASE_PATH, "data", "metadata", "manual_ocr.json")


def process_ocr_task(task):
    """
    Worker function for processing a single OCR asset
    """
    img_path, segmentation, anno_id, img_filename, path_root = task
    
    asset_img = HarvesterCore.surgical_crop(img_path, segmentation)  # Surgical Crop
    if not asset_img:
        return None

    asset_id = f"ocr_{anno_id:04d}"  # Save Asset
    os.makedirs(PATH_OUT_ASSETS, exist_ok=True)
    
    asset_path = os.path.join(PATH_OUT_ASSETS, f"{asset_id}.png")
    asset_img.save(asset_path)

    epoch = HarvesterCore.get_epoch(img_filename)  # Record Metadata
    
    return {
        "asset_id": asset_id,
        "original_file": img_filename,
        "epoch": epoch,
        "path_rel": relpath_posix(asset_path, path_root),
        "category": "printed",
        "polygon": segmentation,
        "dims": {"w": int(asset_img.width), "h": int(asset_img.height)}
    }


def run(workers=4):
    """
    Execute the OCR harvest pipeline in parallel
    """
    print("[*] Loading COCO database")
    coco = HarvesterCore.load_coco()
    if not coco:
        return

    image_lookup = {img["id"]: img for img in coco["images"]}
    cat_lookup = {cat["id"]: cat["name"] for cat in coco["categories"]}

    tasks = []
    print("[*] Planning Printed (OCR) harvest")

    for anno in coco["annotations"]:
        cat_name = cat_lookup.get(anno["category_id"])
        if cat_name != "OCR":
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
    print(f"[*] Harvesting {len(tasks)} OCR assets using {workers} workers")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_ocr_task, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="OCR Harvest"):
            res = future.result()
            if res:
                metadata.append(res)

    os.makedirs(os.path.dirname(PATH_OUT_METADATA), exist_ok=True)
    with open(PATH_OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[*] Printed metadata saved to: {PATH_OUT_METADATA}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel OCR Harvester")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    run(workers=args.workers)
