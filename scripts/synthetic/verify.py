# -*- coding: utf-8 -*-
"""
Smoke test for the Synthetic Data Factory

Mirrors main.py's per-document recipe (prune + cross-template inject +
assemble + save) and triple-fans-out across the three modes: for each
template/seed combination it generates the SAME document under realistic,
synthetic, and hybrid mode so the user can compare side-by-side how each
source mix renders the same layout

Inputs:  layout templates under data/layouts/templates/{epoch}/
Outputs: count documents per mode at data/synthetic/verify/{mode}/,
         each with images, layers, masks and annotations files  Same
         filenames across modes so direct visual comparison is trivial
"""

import os
import math
import json
import argparse
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.synthetic.factory import generate_sample
from src.synthetic.layouts.augment import plan_variants, warn_if_scarce
from src.core.config import EPOCHS as EPOCH_ALL

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIR_TEMPLATES = os.path.join(PATH_ROOT, "data", "layouts", "templates")
PATH_DIR_VERIFY = os.path.join(PATH_ROOT, "data", "synthetic", "factory", "verify")

BASE_SEED  = 42
MODES_ALL  = ["realistic", "synthetic", "hybrid"]


def _list_templates(templates_root, epoch_list):
    """Returns [(epoch, tpl_path)] for every .json template in the requested epoch dirs"""
    result = []
    for epoch in epoch_list:
        epoch_dir = os.path.join(templates_root, epoch)
        if not os.path.isdir(epoch_dir): continue  # Epoch directory absent or empty
        for fname in sorted(os.listdir(epoch_dir)):
            if fname.endswith(".json"):
                result.append((epoch, os.path.join(epoch_dir, fname)))
    return result


def _load_template_regions(path):
    """Reads a template JSON and returns its regions list (empty list when missing)"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("regions", [])


def main(count_arg, epoch_list, workers):
    """Plans a small task batch, fans it out across all three modes, dispatches workers"""
    target_eras = EPOCH_ALL if "all" in epoch_list else epoch_list
    print(f"[*] Verify | Epochs: {target_eras} | Workers: {workers} | Count: {count_arg} | Modes: {MODES_ALL}", flush=True)

    all_templates = _list_templates(PATH_DIR_TEMPLATES, target_eras)
    if not all_templates:
        print("[!] No templates found", flush=True)
        return

    # Build per-epoch donor pool (same logic as main)
    epoch_pools = {}
    for epoch in target_eras:
        donor_regions = []
        for ep, tpl_path in all_templates:
            if ep == epoch:
                donor_regions.extend(_load_template_regions(tpl_path))
        epoch_pools[epoch] = donor_regions

    variants_by_epoch = plan_variants(count_arg, all_templates, target_eras)
    warn_if_scarce(variants_by_epoch)
    per_epoch_target  = (math.ceil(count_arg / len(variants_by_epoch))
                         if count_arg is not None and variants_by_epoch else None)

    # Prepare tasks - fan out each layout decision across all three modes
    # so the same template/seed produces a realistic, synthetic, and hybrid render
    tasks   = []
    doc_idx = 0
    for epoch, v in variants_by_epoch.items():
        epoch_templates = [(e, p) for e, p in all_templates if e == epoch]
        epoch_docs      = 0
        for t_idx, (_, tpl_path) in enumerate(epoch_templates):
            for k in range(v):
                if per_epoch_target is not None and epoch_docs >= per_epoch_target: break
                if count_arg is not None and doc_idx >= count_arg: break
                seed = BASE_SEED * 10000 + doc_idx
                # Same seed for all three modes so prune+inject produces an identical layout;
                # only the content fill (HW/OCR source) differs
                for mode in MODES_ALL:
                    mode_output_root = os.path.join(PATH_DIR_VERIFY, mode)
                    tasks.append((
                        epoch, tpl_path, k, t_idx, seed, mode_output_root, epoch_pools[epoch], mode
                    ))
                doc_idx    += 1
                epoch_docs += 1
            if count_arg is not None and doc_idx >= count_arg: break

    print(f"[*] Dispatching {len(tasks)} verify tasks ({doc_idx} layouts x {len(MODES_ALL)} modes)", flush=True)

    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate_sample, task) for task in tasks]
        for future in as_completed(futures):
            try:
                info = future.result()
                completed += 1
                print(f"[{completed}/{len(tasks)}] Generated {info['name']} ({info['mode']}) | "
                      f"Regions: {info['regions']} | "
                      f"Template: {info['template']}", flush=True)
            except Exception as e:  # Surface worker exceptions but keep draining
                print(f"[!] Error in worker process: {e}", flush=True)

    print(f"\n[*] Verification complete | {completed} renders at {PATH_DIR_VERIFY}/{{realistic,synthetic,hybrid}}/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Factory verify (all modes)")
    parser.add_argument("--count",   type=int, default=8)
    parser.add_argument("--epoch",   type=str, nargs="+", default=["all"])
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    main(args.count, args.epoch, args.workers)
