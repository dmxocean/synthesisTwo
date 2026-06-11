# -*- coding: utf-8 -*-
"""
Zips the cluster-selected PDFs identified by sample_layouts.py

Preserves the raw archive structure: {year}/{filename}.pdf. Reads the DBSCAN
manifest produced by sample_layouts.py, deduplicates by source PDF, and
writes a zip with one entry per unique PDF

Usage:
    python scripts/tools/zip_cluster_pdfs.py
    python scripts/tools/zip_cluster_pdfs.py --epoch monarchy war
    python scripts/tools/zip_cluster_pdfs.py --output data/analysis/subset.zip
"""

import os
import json
import zipfile
import argparse
from tqdm import tqdm

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIR_LAYOUTS_SAMPLES = os.path.join(PATH_ROOT, "data", "interim", "layouts", "samples")
PATH_DIR_ANALYSIS = os.path.join(PATH_ROOT, "data", "analysis")

PATH_MANIFEST_DEFAULT = os.path.join(PATH_DIR_LAYOUTS_SAMPLES, "manifest.json")
PATH_OUTPUT_DEFAULT   = os.path.join(PATH_DIR_ANALYSIS, "cluster_pdfs.zip")


def _strip_prefix(source: str) -> str:
    """Converts a manifest source path to the zip entry path

    data/raw/1925/file.pdf  ->  1925/file.pdf

    Always returns forward slashes - zip entries must be portable across platforms
    """
    norm = os.path.normpath(source)    # Handles / and \ uniformly
    prefix = os.path.normpath(os.path.join("data", "raw"))    # data\raw on Windows
    if norm.startswith(prefix + os.sep):
        rel = norm[len(prefix) + len(os.sep):]    # 1925\file.pdf (Windows)
    else:
        parts = norm.split(os.sep)    # Fallback: take the last two components (year/file)
        rel = os.path.join(*parts[-2:]) if len(parts) >= 2 else os.path.basename(norm)
    return rel.replace(os.sep, "/")


def run(manifest_path: str, output_path: str, epochs: list[str] | None) -> None:
    """Zips the cluster-selected PDFs identified by the manifest
    
    Filters by epoch if requested, deduplicates by source PDF, and writes
    the result to the output zip file
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run scripts/synthetic/sample_layouts.py first"
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if epochs:    # Filter by epoch if requested
        manifest = [e for e in manifest if e["epoch"] in epochs]
        if not manifest:
            print(f"[!] No entries found for epoch(s): {epochs}")
            return

    seen: dict[str, str] = {}    # Deduplicate: source -> epoch
    for entry in manifest:
        src = entry["source"].replace("\\", "/")
        seen[src] = entry["epoch"]

    print(f"[*] Manifest entries : {len(manifest)}")
    print(f"[*] Unique PDFs      : {len(seen)}")

    epoch_counts: dict[str, int] = {}    # Per-epoch breakdown
    for src, epoch in seen.items():
        epoch_counts[epoch] = epoch_counts.get(epoch, 0) + 1
    for ep, n in sorted(epoch_counts.items()):
        print(f"    {ep:<15} {n:>4} PDFs")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    skipped = 0
    zipped  = 0

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source, epoch in tqdm(seen.items(), desc="Zipping PDFs", unit="pdf"):
            abs_path  = os.path.join(PATH_ROOT, source.replace("/", os.sep))
            zip_entry = _strip_prefix(source)

            if not os.path.exists(abs_path):
                print(f"[!] Missing, skipped: {source}")
                skipped += 1
                continue

            zf.write(abs_path, arcname=zip_entry)
            zipped += 1

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"\n[*] Zipped  : {zipped} PDFs")
    if skipped:
        print(f"[!] Skipped : {skipped} PDFs (not found on disk)")
    print(f"[*] Output  : {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip DBSCAN-selected PDFs")
    parser.add_argument(
        "--manifest",
        default=PATH_MANIFEST_DEFAULT,
        help="Path to manifest.json (default: data/interim/layouts/samples/manifest.json)",
    )
    parser.add_argument(
        "--output",
        default=PATH_OUTPUT_DEFAULT,
        help="Destination zip file (default: data/analysis/cluster_pdfs.zip)",
    )
    parser.add_argument(
        "--epoch",
        nargs="+",
        metavar="EPOCH",
        help="Filter to one or more epochs: monarchy republic war francoist",
    )
    args = parser.parse_args()
    run(args.manifest, args.output, args.epoch)
