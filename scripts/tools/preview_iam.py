# -*- coding: utf-8 -*-
"""
Visual style browser for the IAM handwriting library (Restored Naming)

Generates one A4 PNG per letter group (a–r) at 300 DPI showing writer samples
at their RAW NATIVE size to reflect real synthesis output
"""

import os
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIR_IAM_LIB = os.path.join(PATH_ROOT, "data", "iam", "library")
PATH_DIR_IAM_SAMPLES = os.path.join(PATH_ROOT, "data", "iam", "samples")

# Constants
SEED_RANDOM = 42
DPI_PAGE = 300
SIZE_PAGE = (2480, 3508)  # A4 at 300 DPI
VAL_MARGIN_PAGE = 80
SIZE_HEIGHT_HEADER = 90
SIZE_HEIGHT_LABEL = 38
VAL_GAP_ROW = 15
VAL_GAP_WRITER = 60
COLOR_PAPER = (250, 248, 243)
COLOR_SEPARATOR = (210, 207, 200)
COLOR_HEADER = (90, 90, 90)
COLOR_LABEL = (50, 50, 50)


def _collect_groups():
    groups = {}
    if not os.path.isdir(PATH_DIR_IAM_LIB):
        return {}
        
    for prefix in sorted(os.listdir(PATH_DIR_IAM_LIB)):
        if not prefix: continue
        group_key = prefix[0].lower()  # Use the first letter as the group key (a, b, c...)
        d = os.path.join(PATH_DIR_IAM_LIB, prefix, "sentences")
        if not os.path.isdir(d): continue
            
        paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))
        if paths:
            groups.setdefault(group_key, {})[prefix] = paths
    return groups


def _load_raw_asset(path):
    img = Image.open(path).convert("RGBA")
    canvas = Image.new("RGB", img.size, COLOR_PAPER)
    canvas.paste(img, (0, 0), img)
    return canvas


def _render_group(group_name, writers, out_path, seed):
    rng = random.Random(seed)
    W, H = SIZE_PAGE
    ids = sorted(writers.keys())
    content_w = W - 2 * VAL_MARGIN_PAGE
    canvas = Image.new("RGB", (W, H), COLOR_PAPER)
    draw = ImageDraw.Draw(canvas)
    header = f"IAM Preview: Group '{group_name.upper()}' · {len(ids)} writer-prefixes · RAW SCALE (300 DPI)"
    draw.text((VAL_MARGIN_PAGE, VAL_MARGIN_PAGE), header, fill=COLOR_HEADER)
    y = VAL_MARGIN_PAGE + SIZE_HEIGHT_HEADER
    
    for wid in ids:
        if y > H - VAL_MARGIN_PAGE - 120: break
        draw.text((VAL_MARGIN_PAGE, y), f"Prefix: {wid}", fill=COLOR_LABEL)
        y += SIZE_HEIGHT_LABEL + 5
        paths = writers[wid]
        samples = rng.sample(paths, min(2, len(paths)))
        for p in samples:
            try:
                asset = _load_raw_asset(p)
                if asset.width > content_w:
                    asset = asset.crop((0, 0, content_w, asset.height))
                if y + asset.height > H - VAL_MARGIN_PAGE: break
                canvas.paste(asset, (VAL_MARGIN_PAGE, y))
                y += asset.height + VAL_GAP_ROW
            except Exception: continue
        y += VAL_GAP_WRITER
        draw.line([(VAL_MARGIN_PAGE, y - VAL_GAP_WRITER//2), (W - VAL_MARGIN_PAGE, y - VAL_GAP_WRITER//2)], fill=COLOR_SEPARATOR)
    canvas.save(out_path, dpi=(DPI_PAGE, DPI_PAGE))


def run(seed):
    groups = _collect_groups()
    if not groups:
        print("[!] No assets found in library.")
        return
    if not os.path.exists(PATH_DIR_IAM_SAMPLES):
        os.makedirs(PATH_DIR_IAM_SAMPLES)
    for group_name, writers in sorted(groups.items()):
        out_path = os.path.join(PATH_DIR_IAM_SAMPLES, f"{group_name}.png")
        _render_group(group_name, writers, out_path, seed)
        print(f"[*] Generated: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED_RANDOM)
    args = parser.parse_args()
    run(args.seed)
