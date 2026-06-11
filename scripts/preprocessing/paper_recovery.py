# -*- coding: utf-8 -*-
"""
Paper-texture harvesting executable workflow

This script owns the harvesting logic by reading the DBSCAN sampling manifest produced by sample_layouts.py, collecting the unique representative PDFs per epoch, and dispatching ink-removal in parallel. The per-PDF ink-removal worker and the epoch mapping come from src/preprocessing/paper_recovery
"""

import os
import json
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from src.core.config import PATH_DIR_PAPER, EPOCHS
from src.preprocessing.paper_recovery import process_paper_task

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_MANIFEST = os.path.join(BASE_PATH, "data", "interim", "layouts", "samples", "manifest.json")


def plan_tasks(samples_per_epoch):
    """
    Build per-PDF recovery tasks from the DBSCAN sampling manifest

    Reads the representative PDF list produced by sample_layouts.py so that paper textures are harvested only from the clustered pages, not the full archive

    Args:
        samples_per_epoch (int): cap on unique PDFs drawn from each epoch
    Returns:
        list[tuple]: tasks for process_paper_task, each (pdf_path, epoch_name, year, count_idx, output_dir)
    """
    if not os.path.exists(PATH_MANIFEST):
        raise FileNotFoundError(
            f"DBSCAN manifest not found: {PATH_MANIFEST}\n"
            "Run scripts/synthetic/sample_layouts.py first"
        )

    with open(PATH_MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    # Collect unique PDFs per epoch from the manifest
    epoch_pdfs: dict[str, set] = {}
    for entry in manifest:
        epoch = entry["epoch"]
        pdf = os.path.join(BASE_PATH, entry["source"])
        epoch_pdfs.setdefault(epoch, set()).add(pdf)

    tasks = []
    for epoch_name in EPOCHS:
        pdfs = sorted(epoch_pdfs.get(epoch_name, []))
        epoch_output_dir = os.path.join(PATH_DIR_PAPER, epoch_name)
        os.makedirs(epoch_output_dir, exist_ok=True)
        random.shuffle(pdfs)
        for count, pdf in enumerate(pdfs[:samples_per_epoch]):
            year = os.path.basename(os.path.dirname(pdf))
            tasks.append((pdf, epoch_name, year, count, epoch_output_dir))
    return tasks


def harvest(samples_per_epoch=125, workers=4):
    """
    Plan the tasks and run ink removal in parallel writing one PNG per PDF

    Args:
        samples_per_epoch (int): max samples per historical epoch
        workers (int): number of parallel worker processes
    """
    random.seed(42)  # Reproduce epoch and PDF selection

    print("Paper Harvester: recovering textures from DBSCAN-selected PDFs")  # Log texture recovery start

    tasks = plan_tasks(samples_per_epoch)
    print(f"Dispatching {len(tasks)} paper-recovery tasks across {workers} workers")  # Log task dispatch

    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_paper_task, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Paper Harvest"):
            if future.result():
                completed += 1

    print(f"Recovery complete - {completed} samples written to {PATH_DIR_PAPER}")  # Log completion status


def main():
    """
    Entry point for the paper texture harvester
    """
    parser = argparse.ArgumentParser(description="Parallel Paper Texture Harvester")
    parser.add_argument("--samples", type=int, default=125, help="Samples per epoch")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    harvest(samples_per_epoch=args.samples, workers=args.workers)


if __name__ == "__main__":
    main()
