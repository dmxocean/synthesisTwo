# -*- coding: utf-8 -*-
"""
Synthetic data generation (executable workflow)

This script owns the generation logic: it discovers templates, builds per-epoch
donor pools, and dispatches document-generation tasks in parallel under one of
two policies:

  --tiles N   target N valid 768x768 crops total, balanced across epochs
              (self-replenishing: new tasks are submitted per epoch only while
              that epoch is below quota)
  --count N   generate exactly N document pages regardless of crop yield

The per-document worker (generate_sample) and template helpers come from
src/synthetic/factory; variant planning comes from src/synthetic/layouts/augment

Run from the project root:
  python scripts/synthetic/generate.py --tiles 2000
  python scripts/synthetic/generate.py --count 100 --epoch war francoist --mode hybrid
"""

import os
import math
import argparse
import random
from itertools import cycle
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.synthetic.factory import generate_sample, _list_templates, _load_template, EPOCH_ALL
from src.synthetic.layouts.augment import plan_variants, warn_if_scarce

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIR_TEMPLATES = os.path.join(PATH_ROOT, "data", "layouts", "templates")
PATH_DIR_OUTPUT = os.path.join(PATH_ROOT, "data", "synthetic", "factory")

COUNT_TILES_DEFAULT = 2000


def _trim_to_quota(target_eras, quota, output_root):
    """
    Delete excess tile files so each epoch keeps at most quota tiles

    Tiles are sorted alphabetically - the zero-padded doc_idx in the filename
    preserves generation order, so the first quota tiles per epoch are kept
    and any overshoot from in-flight workers is removed

    Removes the matching label TIFF and annotation JSON alongside each
    deleted image so all three output directories stay in sync

    Returns:
        int: total tiles kept across all epochs
    """
    dir_img = os.path.join(output_root, "images")
    dir_lbl = os.path.join(output_root, "labels")
    dir_ann = os.path.join(output_root, "annotations")
    total_kept = 0

    for epoch in target_eras:
        prefix = f"synth_{epoch}_"
        tiles  = sorted(
            f for f in os.listdir(dir_img)
            if f.startswith(prefix) and f.endswith("_input.png")
        )
        keep   = tiles[:quota]
        excess = tiles[quota:]
        total_kept += len(keep)

        for fname in excess:
            base = fname[: -len("_input.png")]
            os.remove(os.path.join(dir_img, fname))
            for path in (
                os.path.join(dir_lbl, f"{base}_label.tiff"),
                os.path.join(dir_ann, f"{base}.json"),
            ):
                if os.path.exists(path):
                    os.remove(path)

        if excess:
            print(f"[*] Trim | epoch={epoch} kept={len(keep)} removed={len(excess)}", flush=True)

    return total_kept


def _run_tile_quota(target_tiles, target_eras, all_templates, epoch_pools, workers, mode, base_seed):
    """
    Generates pages until each epoch accumulates its fair share of valid tiles

    Then trims any overshoot so the final count is exactly target_tiles

    Returns:
        int: total tiles saved after trimming
    """
    quota = math.ceil(target_tiles / len(target_eras))
    tiles_done = {e: 0 for e in target_eras}
    tpl_iters = {e: cycle([(ep, p) for ep, p in all_templates if ep == e]) for e in target_eras}

    doc_idx = 0
    total_tiles = 0
    completed = 0

    def _next_task(epoch):
        nonlocal doc_idx
        _, tpl_path = next(tpl_iters[epoch])
        seed = (base_seed * 10000 + doc_idx) % (2 ** 32)
        task = (epoch, tpl_path, doc_idx, doc_idx, seed, PATH_DIR_OUTPUT, epoch_pools[epoch], mode)
        doc_idx += 1
        return task

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {}
        slots_per_epoch = max(1, workers // len(target_eras))
        for epoch in target_eras:
            for _ in range(slots_per_epoch):
                if tiles_done[epoch] < quota:
                    pending[executor.submit(generate_sample, _next_task(epoch))] = epoch

        while pending:
            for future in list(as_completed(pending)):
                epoch = pending.pop(future)
                try:
                    info = future.result()
                except Exception as e:
                    print(f"[!] Worker error: {e}", flush=True)
                    if tiles_done[epoch] < quota:
                        pending[executor.submit(generate_sample, _next_task(epoch))] = epoch
                    continue

                tiles_done[epoch] += info["tiles"]
                total_tiles += info["tiles"]
                completed += 1
                print(f"[{completed}] {info['name']} | tiles={info['tiles']} | "
                      f"{epoch}={tiles_done[epoch]}/{quota}", flush=True)

                if tiles_done[epoch] < quota:
                    pending[executor.submit(generate_sample, _next_task(epoch))] = epoch

    summary = " ".join(f"{e}={tiles_done[e]}" for e in target_eras)
    print(f"[*] Tile quota complete | raw={total_tiles} | {summary}", flush=True)

    total_kept = _trim_to_quota(target_eras, quota, PATH_DIR_OUTPUT)
    print(f"[*] After trim | exact={total_kept}", flush=True)
    return total_kept


def _run_page_count(count_arg, target_eras, all_templates, epoch_pools, workers, mode, base_seed):
    """
    Legacy mode: generate exactly count_arg document pages

    Returns:
        int: total tiles saved
    """
    variants_by_epoch = plan_variants(count_arg, all_templates, target_eras)
    warn_if_scarce(variants_by_epoch)
    per_epoch_target = (math.ceil(count_arg / len(variants_by_epoch)) if variants_by_epoch else None)

    tasks = []
    doc_idx = 0
    for epoch, v in variants_by_epoch.items():
        epoch_templates = [(e, p) for e, p in all_templates if e == epoch]
        epoch_docs = 0
        for t_idx, (_, tpl_path) in enumerate(epoch_templates):
            for k in range(v):
                if per_epoch_target is not None and epoch_docs >= per_epoch_target:
                    break
                if count_arg is not None and doc_idx >= count_arg:
                    break
                seed = (base_seed * 10000 + doc_idx) % (2 ** 32)
                tasks.append((epoch, tpl_path, k, t_idx, seed, PATH_DIR_OUTPUT, epoch_pools[epoch], mode))
                doc_idx += 1
                epoch_docs += 1
            if count_arg is not None and doc_idx >= count_arg:
                break

    print(f"[*] Dispatching {len(tasks)} page tasks...", flush=True)
    completed_count = 0
    total_tiles = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate_sample, task) for task in tasks]
        for future in as_completed(futures):
            try:
                info = future.result()
                completed_count += 1
                total_tiles += info["tiles"]
                print(f"[{completed_count}/{len(tasks)}] {info['name']} | "
                      f"tiles={info['tiles']} | template={info['template']}", flush=True)
            except Exception as e:
                print(f"[!] Worker error: {e}", flush=True)
    return total_tiles


def main(count_arg, tiles_arg, epoch_list, workers, mode, seed):
    """Plans and dispatches synthetic document generation"""
    base_seed = seed if seed is not None else random.randint(0, 2 ** 31 - 1)
    target_eras = EPOCH_ALL if "all" in epoch_list else epoch_list
    print(f"[*] Factory | epochs={target_eras} workers={workers} mode={mode} seed={base_seed}", flush=True)

    all_templates = _list_templates(PATH_DIR_TEMPLATES, target_eras)
    if not all_templates:
        print("[!] No templates found.", flush=True)
        return

    epoch_pools = {}  # Per-epoch donor pool from all templates of that epoch
    for epoch in target_eras:
        donor_regions = []
        for ep, tpl_path in all_templates:
            if ep == epoch:
                donor_regions.extend(_load_template(tpl_path).get("regions", []))
        epoch_pools[epoch] = donor_regions

    if tiles_arg is not None:
        print(f"[*] Tile-quota mode | target={tiles_arg} crops "
              f"(~{math.ceil(tiles_arg / len(target_eras))} per epoch)", flush=True)
        total = _run_tile_quota(tiles_arg, target_eras, all_templates, epoch_pools, workers, mode, base_seed)
    else:
        total = _run_page_count(count_arg, target_eras, all_templates, epoch_pools, workers, mode, base_seed)

    print(f"[*] Synthesis complete | {total} total tiles written to {PATH_DIR_OUTPUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Synthetic Data Factory")
    parser.add_argument("--tiles",   type=int, default=COUNT_TILES_DEFAULT,
                        help="target number of valid 768x768 crops (tile-quota mode)")
    parser.add_argument("--count",   type=int, default=None,
                        help="generate exactly N pages instead of targeting tiles (legacy mode)")
    parser.add_argument("--epoch",   type=str, nargs="+", default=["all"])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--mode",    type=str, choices=["realistic", "synthetic", "hybrid"], default="hybrid")
    parser.add_argument("--seed",    type=int, default=42,
                        help="base random seed; omit for a fresh random seed each run")
    args = parser.parse_args()
    tiles_arg = None if args.count is not None else args.tiles
    main(args.count, tiles_arg, args.epoch, args.workers, args.mode, args.seed)
