# -*- coding: utf-8 -*-
"""
Convert DELINE8K 5-frame labels to the factory 4-frame binary format, in place

DELINE8K frames: 0=noise, 1=handwritten, 2=empty (drop), 3=printed, 4=intersection
Factory frames : 0=ns, 1=hw, 2=pr, 3=inter, binary {0,255} (the dataset reads 0,1,2)
So we keep ch0->ns, ch1->hw, ch3->pr, binarize, and recompute the intersection

Idempotent (skips labels already 4-frame) and atomic (writes a temp file then replaces)

  python scripts/tools/convert_deline8k_labels.py --dry-run
  python scripts/tools/convert_deline8k_labels.py
"""

import os
import glob
import argparse

import numpy as np
from PIL import Image, ImageSequence

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LBL_DIR = os.path.join(PATH_ROOT, "data", "synthetic", "deline8k", "labels")

SRC = {"ns": 0, "hw": 1, "pr": 3}  # DELINE8K source frame for each project channel


def _frames(path):
    return [np.array(fr.convert("L")) for fr in ImageSequence.Iterator(Image.open(path))]


def convert_one(path):
    """Returns 'converted' | 'skip' | 'bad' for one label file"""
    frames = _frames(path)
    if len(frames) == 4:
        return "skip"  # Already in factory format
    if len(frames) <= SRC["pr"]:
        return "bad"   # Not the expected 5-frame DELINE8K layout
    ns = (frames[SRC["ns"]] > 0).astype(np.uint8) * 255
    hw = (frames[SRC["hw"]] > 0).astype(np.uint8) * 255
    pr = (frames[SRC["pr"]] > 0).astype(np.uint8) * 255
    inter = (((ns > 0) & (hw > 0)) | ((ns > 0) & (pr > 0)) | ((hw > 0) & (pr > 0))).astype(np.uint8) * 255
    out = [Image.fromarray(x, mode="L") for x in (ns, hw, pr, inter)]
    tmp = path + ".tmp.tiff"
    out[0].save(tmp, format="TIFF", save_all=True, append_images=out[1:], compression="tiff_deflate")
    os.replace(tmp, path)
    return "converted"


def main(args):
    labels = sorted(glob.glob(os.path.join(args.labels, "*_label.tiff")))
    if args.limit:
        labels = labels[: args.limit]
    print(f"[*] {len(labels)} labels under {args.labels}", flush=True)

    counts = {"converted": 0, "skip": 0, "bad": 0}
    for i, p in enumerate(labels):
        if args.dry_run:
            n = Image.open(p).n_frames
            print(f"  {os.path.basename(p)}: {n} frames -> {'convert' if n != 4 else 'skip'}")
            continue
        counts[convert_one(p)] += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(labels)}", flush=True)
    if not args.dry_run:
        print(f"[*] converted={counts['converted']} skipped={counts['skip']} bad={counts['bad']}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Standardize DELINE8K labels to the factory 4-frame binary format")
    p.add_argument("--labels", default=LBL_DIR)
    p.add_argument("--limit", type=int, default=0, help="Cap files (for a test run)")
    p.add_argument("--dry-run", action="store_true")
    main(p.parse_args())
