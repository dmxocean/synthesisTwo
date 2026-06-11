# -*- coding: utf-8 -*-
"""
Kernel visualization and region extraction analysis

This script evaluates different RLSA (Run-Length Smoothing Algorithm) kernels for region extraction. It generates visual overlays of detected regions and calculates quantitative metrics like fragmentation and coverage to help select optimal kernels for the production pipeline
"""

import os
import sys
import csv

import cv2
import numpy as np
from PIL import Image

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_PATH)

from src.preprocessing.pdf import DocumentProcessor
from src.preprocessing.normalize import preprocess_ctr_bgd_gs
from src.segmentation.inference import load_segmenter, predict_layers
from src.detection.region.extract import extract_regions
from src.core.gpu import DeviceManager

TEST_PDFS = [
    ("data/raw/1938/guiradbcn_a1938m1.pdf", 0),
    ("data/raw/1925/guiradbcn_a1925m2.pdf", 0),
    ("data/raw/1938/guiradbcn_a1938m5.pdf", 0),
    ("data/raw/1945/guiradbcn_a1945m6d10.pdf", 0),
    ("data/raw/1950/guiradbcn_a1950m1d2.pdf", 0),
]


def _region_metrics(regs, page_area):
    """
    Compute quantifiable metrics for a list of regions
    """
    if not regs:
        return {"count": 0, "mean_area": 0, "median_area": 0,
                "fragmentation": 0, "coverage_pct": 0}
    areas = [r.bbox[2] * r.bbox[3] for r in regs]
    mean_a  = float(np.mean(areas))
    std_a   = float(np.std(areas))
    return {
        "count":          len(regs),
        "mean_area":      round(mean_a),
        "median_area":    round(float(np.median(areas))),
        "fragmentation":  round(std_a / mean_a if mean_a > 0 else 0, 3),
        "coverage_pct":   round(sum(areas) / page_area * 100, 2),
    }


def visualize_extraction(pdf_path, page_idx, model, device, kernels, output_dir):
    """
    Execute region extraction and visualization for a set of kernels

    Args:
        pdf_path (str): Path to the source PDF file
        page_idx (int): Zero-based index of the page to process
        model (torch.nn.Module): Loaded segmentation model
        device (torch.device): Target compute device
        kernels (List[Tuple]): Collection of (width, height) RLSA kernels to evaluate
        output_dir (str): Directory for visual outputs
    Returns:
        List[Dict]: Collection of extraction metrics for each kernel
    """
    name = os.path.basename(pdf_path).replace(".pdf", "")
    print(f"[*] Visualizing {name}")

    img = DocumentProcessor.pdf_page_to_image(pdf_path, page_index=page_idx)
    if img is None:
        return []

    inp, _ = preprocess_ctr_bgd_gs(img)
    probs  = predict_layers(model, inp, device)
    H, W   = img.shape[:2]
    page_area = H * W

    pdf_dir = os.path.join(output_dir, name)
    os.makedirs(pdf_dir, exist_ok=True)

    rows = []
    for k_rlsa in kernels:
        ns_regs = extract_regions(probs[0], prefix="ns", kernel=k_rlsa)
        hw_regs = extract_regions(probs[1], prefix="hw", kernel=k_rlsa)
        pr_regs = extract_regions(probs[2], prefix="pr", kernel=k_rlsa)

        canvas = img.copy()  # Rendering visual overlays
        for r in ns_regs:
            x, y, w, h = r.bbox
            cv2.rectangle(canvas, (x, y), (x+w, y+h), (255, 150, 0), 3)
            cv2.putText(canvas, r.region_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 0), 2)
        for r in hw_regs:
            x, y, w, h = r.bbox
            cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(canvas, r.region_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        for r in pr_regs:
            x, y, w, h = r.bbox
            cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(canvas, r.region_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out_path = os.path.join(pdf_dir, f"k{k_rlsa[0]}x{k_rlsa[1]}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        pr_m = _region_metrics(pr_regs, page_area)
        hw_m = _region_metrics(hw_regs, page_area)
        ns_m = _region_metrics(ns_regs, page_area)

        row = {
            "pdf_id":             name,
            "kernel_w":           k_rlsa[0],
            "kernel_h":           k_rlsa[1],
            "pr_count":           pr_m["count"],
            "pr_mean_area":       pr_m["mean_area"],
            "pr_median_area":     pr_m["median_area"],
            "pr_fragmentation":   pr_m["fragmentation"],
            "pr_coverage_pct":    pr_m["coverage_pct"],
            "hw_count":           hw_m["count"],
            "hw_mean_area":       hw_m["mean_area"],
            "hw_fragmentation":   hw_m["fragmentation"],
            "hw_coverage_pct":    hw_m["coverage_pct"],
            "ns_count":           ns_m["count"],
            "ns_mean_area":       ns_m["mean_area"],
            "ns_fragmentation":   ns_m["fragmentation"],
            "ns_coverage_pct":    ns_m["coverage_pct"],
        }
        rows.append(row)

        print(f"  k{k_rlsa[0]}x{k_rlsa[1]:>3}  "
              f"PR: {pr_m['count']:>3} regs  area={pr_m['mean_area']:>7,}  "
              f"frag={pr_m['fragmentation']:.2f}  cov={pr_m['coverage_pct']:.1f}%  |  "
              f"HW: {hw_m['count']:>3}  NS: {ns_m['count']:>3}")

    return rows


if __name__ == "__main__":
    device = DeviceManager.get_device()
    model  = load_segmenter("unet", device)

    kernels = [(25, 9), (50, 80), (100, 150)]

    output_dir = os.path.join(BASE_PATH, "outputs", "debug", "kernels")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("[*] Kernel Visualization + Metrics")
    print(f"    PDFs: {len(TEST_PDFS)}")
    print(f"    Kernels: {kernels}")
    print(f"{'='*80}\n")

    all_rows = []
    for pdf_rel, page_idx in TEST_PDFS:
        pdf_abs = os.path.join(BASE_PATH, pdf_rel)
        if os.path.exists(pdf_abs):
            rows = visualize_extraction(pdf_abs, page_idx, model, device, kernels, output_dir)
            all_rows.extend(rows)
        else:
            print(f"[!] Path not found: {pdf_abs}")

    csv_path = os.path.join(output_dir, "kernel_metrics.csv")
    fieldnames = [
        "pdf_id", "kernel_w", "kernel_h",
        "pr_count", "pr_mean_area", "pr_median_area", "pr_fragmentation", "pr_coverage_pct",
        "hw_count", "hw_mean_area", "hw_fragmentation", "hw_coverage_pct",
        "ns_count", "ns_mean_area", "ns_fragmentation", "ns_coverage_pct",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n[✓] kernel_metrics.csv → {csv_path}")
