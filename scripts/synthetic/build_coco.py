# -*- coding: utf-8 -*-
"""
COCO instance segmentation JSON builder for synthetic document data

Reads the per-image annotation JSONs produced by DataExporter and assembles a
single coco_instances.json at the synthetic data root

GT polygons are taken directly from the 'gt_polygon' field stored in each
annotation - no mask PNG files are read and no cv2.findContours extraction is
performed  This means the COCO segmentation polygons match the original
Roboflow annotations exactly

Seven fine-grained categories are emitted, grouped under three supercategories:
  HTR   -> handwritten
  OCR   -> printed
  noise -> circles | lines | crosses | marks | stamps
Models can train on all 7 or collapse to 3 via supercategory
"""

import os
import glob
import json
import numpy as np
from PIL import Image, ImageDraw

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIR_OUTPUT = os.path.join(PATH_ROOT, "data", "synthetic", "factory")

CATEGORIES = [
    {"id": 1, "name": "handwritten", "supercategory": "HTR"},
    {"id": 2, "name": "printed",     "supercategory": "OCR"},
    {"id": 3, "name": "circles",     "supercategory": "noise"},
    {"id": 4, "name": "lines",       "supercategory": "noise"},
    {"id": 5, "name": "crosses",     "supercategory": "noise"},
    {"id": 6, "name": "marks",       "supercategory": "noise"},
    {"id": 7, "name": "stamps",      "supercategory": "noise"},
]

_CATEGORY_ID = {  # Maps annotation category + subtype to COCO category_id
    "handwritten": 1,
    "printed":     2,
    "circles":     3,
    "lines":       4,
    "crosses":     5,
    "marks":       6,
    "stamps":      7,
}


def _polygon_area(gt_polygon, img_w, img_h):
    """
    Computes the pixel area of a gt_polygon by rasterising it into a temporary mask
    Returns 0 if the polygon is empty or invalid
    """
    if not gt_polygon:
        return 0
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    for part in gt_polygon:
        pts = [(int(part[i]), int(part[i + 1])) for i in range(0, len(part) - 1, 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return int(np.array(mask).sum() // 255)


def _resolve_category_id(ann):
    """
    Resolves the COCO category_id from an annotation dict
    Uses 'subtype' for noise sub-types, otherwise 'category'
    Returns None if the annotation cannot be mapped
    """
    subtype = ann.get("subtype")
    if subtype and subtype in _CATEGORY_ID:
        return _CATEGORY_ID[subtype]
    category = ann.get("category", "")
    return _CATEGORY_ID.get(category)


def build_coco_dataset(path_synthetic_root):
    """
    Assembles coco_instances.json from all per-image annotation JSONs

    Scans annotations/ for *.json files, reads each annotation's gt_polygon
    directly as the COCO segmentation field, and writes one coco_instances.json
    at path_synthetic_root

    Args:
        path_synthetic_root (str): path to data/synthetic/

    Returns:
        str: absolute path of the written JSON file
    """
    path_anno_dir = os.path.join(path_synthetic_root, "annotations")
    anno_files    = sorted(glob.glob(os.path.join(path_anno_dir, "*.json")))
    print(f"[*] COCO build: {len(anno_files)} annotation files found")

    images      = []
    annotations = []
    image_id    = 0
    anno_id     = 0

    for anno_file in anno_files:
        with open(anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data["width"]
        img_h = data["height"]

        images.append({
            "id":        image_id,
            "file_name": data["image"],
            "width":     img_w,
            "height":    img_h,
        })

        for ann in data.get("annotations", []):
            cat_id = _resolve_category_id(ann)
            if cat_id is None:
                continue

            bbox       = [int(v) for v in ann["bbox"]]
            gt_polygon = ann.get("gt_polygon")

            if gt_polygon:
                segmentation = gt_polygon
                area         = _polygon_area(gt_polygon, img_w, img_h)
            else:
                bx, by, bw, bh = bbox  # Fallback: encode bbox as a rectangle polygon
                segmentation = [[bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]]
                area         = bw * bh

            if area == 0:
                continue

            annotations.append({
                "id":           anno_id,
                "image_id":     image_id,
                "category_id":  cat_id,
                "segmentation": segmentation,
                "bbox":         bbox,
                "area":         area,
                "iscrowd":      0,
            })
            anno_id += 1

        image_id += 1
        if image_id % 100 == 0:
            print(f"[*] COCO build: {image_id}/{len(anno_files)} images processed")

    coco = {"categories": CATEGORIES, "images": images, "annotations": annotations}
    out_path = os.path.join(path_synthetic_root, "coco_instances.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"[*] COCO JSON: {len(images)} images, {len(annotations)} annotations -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build COCO instance JSON from synthetic data")
    parser.add_argument("--input", default=PATH_DIR_OUTPUT, help="Path to data/synthetic/")
    args = parser.parse_args()
    build_coco_dataset(args.input)
