# -*- coding: utf-8 -*-
"""
Temporary GT polygon inspector

Draws only the raw gt_polygon from each annotation JSON over its page, with
nothing else applied - no morphology, no badges, no class colours, no
template overlay. Use this to spot-check that the polygons stored in the
annotations file genuinely match what the visualisation panels claim

Each vertex is drawn as a small dot so the user can count vertices visually
and see exactly which points define the polygon. Output is written next to
the source annotations under a sibling raw_gt/ folder

Invocation:
  python scripts/tools/show_gt_polygons.py \
      --root data/synthetic/verify/polygons/hybrid
"""

import os
import json
import glob
import argparse
from PIL import Image, ImageDraw

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tuning
COLOR_OUTLINE = (0, 220, 0)    # Bright green outline
COLOR_VERTEX  = (255, 0, 0)    # Red dot per vertex
WIDTH_OUTLINE = 3
RADIUS_VERTEX = 5


def _draw_gt_polygon(draw, polygon):
    """Draws every polygon part as outline + vertex dots, no fill

    Each vertex is drawn as a small dot so the user can count vertices visually
    and see exactly which points define the polygon

    Args:
        draw    (ImageDraw.Draw): pillow drawing context bound to the page
        polygon (list):           COCO-style [[x1,y1,x2,y2,...], ...]
    """
    for part in polygon:
        pts = [(int(part[i]), int(part[i + 1])) for i in range(0, len(part) - 1, 2)]
        if len(pts) < 2:
            continue
        draw.line(pts + [pts[0]], fill=COLOR_OUTLINE, width=WIDTH_OUTLINE)
        for (x, y) in pts:
            draw.ellipse([x - RADIUS_VERTEX, y - RADIUS_VERTEX,
                          x + RADIUS_VERTEX, y + RADIUS_VERTEX],
                         fill=COLOR_VERTEX, outline=(0, 0, 0))


def main(root):
    """Loads every annotations JSON under root and emits one raw GT overlay per page

    Args:
        root (str): directory containing images/ and annotations/ subfolders
    """
    dir_img  = os.path.join(root, "images")
    dir_anno = os.path.join(root, "annotations")
    dir_out  = os.path.join(root, "raw_gt")
    os.makedirs(dir_out, exist_ok=True)

    anno_files = sorted(glob.glob(os.path.join(dir_anno, "*.json")))
    if not anno_files:
        print(f"[!] No annotations under {dir_anno}", flush=True)
        return

    for anno_path in anno_files:
        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_path = os.path.join(dir_img, data["image"])
        if not os.path.exists(img_path):
            print(f"[!] Missing page image {img_path}", flush=True)
            continue

        page = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(page)

        total_polys = 0
        total_verts = 0
        for ann in data.get("annotations", []):
            gt_poly = ann.get("gt_polygon")
            if not gt_poly:
                continue
            _draw_gt_polygon(draw, gt_poly)
            total_polys += len(gt_poly)
            total_verts += sum(len(p) // 2 for p in gt_poly)

        stem = os.path.splitext(data["image"])[0]
        out_path = os.path.join(dir_out, f"{stem}.gt_only.png")
        page.save(out_path)
        print(f"[*] {stem}: {total_polys} polygon parts, {total_verts} total vertices written to {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw raw gt_polygon vertices over each page")
    parser.add_argument("--root", required=True,
                        help="directory containing images/ and annotations/ subfolders, e.g. "
                             "data/synthetic/verify/polygons/hybrid")
    args = parser.parse_args()
    main(args.root)
