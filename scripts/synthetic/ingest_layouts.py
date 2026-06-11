# -*- coding: utf-8 -*-
"""
Roboflow COCO Ingestor for Layout Templates

Converts a Roboflow COCO instance segmentation export into the template JSON
format expected by DocumentAssembler  One template JSON is written per
annotated image, saved under data/layouts/templates/{epoch}/

Class mapping from Roboflow annotation classes to assembler region types:
    HTR                          -> handwritten_region
    OCR                          -> printed_region
    Circles / Lines / Crosses
    Marks / Stamps               -> noise_region

Epoch is inferred from the year embedded in the source filename using the
same year→epoch table as the rest of the pipeline
"""

import json
import os
import re
import yaml
from src.core.config import DICT_EPOCHS

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_COCO = os.path.join(PATH_ROOT, "data", "interim", "layouts", "annotations", "train", "_annotations.coco.json")
PATH_OUTPUT = os.path.join(PATH_ROOT, "data", "layouts", "templates")

ROBOFLOW_TO_REGION = {  # All Roboflow class names -> assembler region type
    "HTR":      "handwritten_region",
    "OCR":      "printed_region",
    "Circles":  "circles_region",
    "Lines":    "lines_region",
    "Crosses":  "crosses_region",
    "Marks":    "marks_region",
    "Crossout": "marks_region",
    "Stamps":   "stamps_region",
}

# Ordered list of noise region types derived from the mapping above (used for classes.yaml)
_NOISE_REGION_TYPES = [
    v for v in ROBOFLOW_TO_REGION.values()
    if v not in ("handwritten_region", "printed_region")
]


def _epoch_from_filename(filename):
    """
    Extracts the year from a Roboflow filename and returns the epoch

    Expected pattern: ...a{year}... e.g. guiradbcn_a1925m7__p002_png.rf.xxxx.png
    Returns None if the year is absent or falls outside all known epochs
    """
    match = re.search(r"_a(\d{4})", filename)
    if not match:
        return None
    year = int(match.group(1))
    for epoch, year_range in DICT_EPOCHS.items():
        if year in year_range:
            return epoch
    return None


def _build_category_map(categories):
    """
    Builds {category_id: region_type} from the COCO categories list

    Unknown Roboflow class names are silently ignored so that future
    additions to the Roboflow project don't crash the ingestor
    """
    mapping = {}
    for cat in categories:
        region_type = ROBOFLOW_TO_REGION.get(cat["name"])
        if region_type:
            mapping[cat["id"]] = region_type
    return mapping


def _polygon_from_annotation(anno):
    """
    Returns the segmentation polygon list from a COCO annotation

    COCO segmentation is [[x1,y1,x2,y2,...], ...] (one list per polygon part)
    Crowd annotations use RLE instead - those are skipped (return None)
    """
    if anno.get("iscrowd", 0):
        return None
    seg = anno.get("segmentation")
    if not seg or not isinstance(seg, list) or not isinstance(seg[0], list):
        return None
    valid = [part for part in seg if len(part) >= 6]  # Keep only polygon parts with at least 3 points
    return valid if valid else None


def _write_classes_yaml(templates_root):
    """
    Scans all templates under templates_root and writes classes.yaml with per-layer coverage

    The YAML records the 3 fixed layers (HTR, OCR, noise) and all noise sub-types derived
    from ROBOFLOW_TO_REGION  Each entry includes how many templates contain at least one
    region of that type, making annotation gaps immediately visible

    Args:
        templates_root (str): root directory containing epoch subdirectories of template JSONs
    """
    type_counts = {}
    for entry in os.listdir(templates_root):
        epoch_path = os.path.join(templates_root, entry)
        if not os.path.isdir(epoch_path):
            continue
        for fname in os.listdir(epoch_path):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(epoch_path, fname), "r", encoding="utf-8") as fh:
                tpl = json.load(fh)
            for rtype in set(r["type"] for r in tpl.get("regions", [])):
                type_counts[rtype] = type_counts.get(rtype, 0) + 1

    data = {
        "layers": [
            {
                "name": "HTR",
                "mask_suffix": "hw",
                "region_type": "handwritten_region",
                "templates_with": type_counts.get("handwritten_region", 0),
            },
            {
                "name": "OCR",
                "mask_suffix": "pr",
                "region_type": "printed_region",
                "templates_with": type_counts.get("printed_region", 0),
            },
            {
                "name": "noise",
                "mask_suffix": "ns",
                "subtypes": [
                    {
                        "name": rtype.replace("_region", ""),
                        "region_type": rtype,
                        "templates_with": type_counts.get(rtype, 0),
                    }
                    for rtype in _NOISE_REGION_TYPES
                ],
            },
        ]
    }

    out_path = os.path.join(templates_root, "classes.yaml")
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"[*] Classes manifest written to {out_path}")


def main():
    if not os.path.exists(PATH_COCO):
        print(f"[!] COCO file not found: {PATH_COCO}")
        return

    print(f"[*] Loading annotations from {PATH_COCO}")
    with open(PATH_COCO, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_map = _build_category_map(data["categories"])  # Build category map dynamically from this export's category IDs
    if not cat_map:
        print("[!] No recognised Roboflow classes found in categories. Check ROBOFLOW_TO_REGION.")
        return

    annos_by_img = {}  # Index annotations by image_id
    for anno in data["annotations"]:
        annos_by_img.setdefault(anno["image_id"], []).append(anno)

    count_ok = count_skip = 0

    for img in data["images"]:
        img_id   = img["id"]
        filename = img["file_name"]

        epoch = _epoch_from_filename(filename)
        if not epoch:
            print(f"[!] Cannot determine epoch for {filename}, skipping.")
            count_skip += 1
            continue

        regions = []
        for anno in annos_by_img.get(img_id, []):
            region_type = cat_map.get(anno["category_id"])
            if not region_type:
                continue

            bbox = [int(float(v)) for v in anno["bbox"]]  # COCO bbox: [x, y, width, height] - convert all to int
            region = {"type": region_type, "bbox": bbox}

            # Preserve the polygon as ground-truth reference (GT only, not a rendering constraint)
            polygon = _polygon_from_annotation(anno)
            if polygon:
                region["gt_polygon"] = polygon

            regions.append(region)

        if not regions:
            print(f"[!] No valid regions for {filename}, skipping.")
            count_skip += 1
            continue

        clean_name = re.sub(r"_png\.rf\..+$", "", os.path.splitext(filename)[0])  # Strip Roboflow hash suffix

        template = {
            "epoch":   epoch,
            "source":  clean_name,
            "width":   img["width"],
            "height":  img["height"],
            "regions": regions,
        }

        epoch_dir = os.path.join(PATH_OUTPUT, epoch)
        os.makedirs(epoch_dir, exist_ok=True)

        out_path = os.path.join(epoch_dir, f"{clean_name}.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

        count_ok += 1

    print(f"[*] Ingested {count_ok} templates to {PATH_OUTPUT}  ({count_skip} skipped)")
    _write_classes_yaml(PATH_OUTPUT)


if __name__ == "__main__":
    main()
