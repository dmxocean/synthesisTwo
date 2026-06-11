# -*- coding: utf-8 -*-
"""
Hyperparameter sweep for preprocessing techniques

This script evaluates various combinations of contrast adjustment, background normalization, and Gaussian denoising against the segmentation model. It generates visual results and a metrics CSV to identify the most effective preprocessing pipeline for different historical document types
"""

import os
import sys
import csv

import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_PATH)

from src.preprocessing.pdf import DocumentProcessor
from src.preprocessing.normalize import normalise_background
from src.segmentation.inference import load_segmenter, predict_layers, COLORS
from src.core.gpu import DeviceManager

THRESHOLDS = [0.2, 0.3, 0.4]

TEST_PDFS = [
    ("data/raw/1938/guiradbcn_a1938m1.pdf", 0),
    ("data/raw/1925/guiradbcn_a1925m2.pdf", 0),
    ("data/raw/1938/guiradbcn_a1938m5.pdf", 0),
    ("data/raw/1945/guiradbcn_a1945m6d10.pdf", 0),
    ("data/raw/1950/guiradbcn_a1950m1d2.pdf", 0),
]


def apply_contrast(img, gamma=1.8):
    """
    Apply gamma-based contrast adjustment
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_gaussian(img, sigma=1.5):
    """
    Apply Gaussian blur for noise reduction
    """
    return cv2.GaussianBlur(img, (0, 0), sigma)


def apply_bg_deletion(img, sigma=100):
    """
    Level background illumination via division by large-sigma Gaussian blur
    """
    img_f = img.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(img_f, (0, 0), sigma)
    norm = img_f / (blur + 1e-6)
    return (np.clip(norm, 0, 1) * 255).astype(np.uint8)


def _norm(img):
    """
    Execute mandatory background normalization
    """
    return normalise_background(img)[0]


def build_techniques():
    """
    Generate ordered preprocessing technique combinations for sweep analysis

    Returns:
        List[Tuple]: Collection of technique names and corresponding lambda functions
    """
    techs = [("norm_only", _norm)]  # Baseline: background normalization only

    ctr_values = [1.5, 2.0, 2.5]

    for ctr in ctr_values:
        techs.append((f"norm_ctr_{ctr}",
                      lambda x, _c=ctr: apply_contrast(_norm(x), _c)))

    for ctr in ctr_values:
        techs.append((f"norm_ctr_{ctr}_bgd_100",
                      lambda x, _c=ctr: apply_bg_deletion(apply_contrast(_norm(x), _c), 100)))

    for ctr in ctr_values:
        for gs in [0.5, 1.0, 1.5, 2.0]:
            techs.append((
                f"norm_ctr_{ctr}_bgd_100_gs_{gs}",
                lambda x, _c=ctr, _gs=gs: apply_gaussian(
                    apply_bg_deletion(apply_contrast(_norm(x), _c), 100), _gs)
            ))

    return techs


def _run_pdf_sweep(pdf_path, page_idx, output_root):
    """
    Execute preprocessing sweep for a single PDF page

    Args:
        pdf_path (str): Path to the source PDF file
        page_idx (int): Zero-based index of the page to process
        output_root (str): Root directory for sweep outputs
    Returns:
        List[Dict]: Collection of pixel metrics for each technique combination
    """
    device = DeviceManager.get_device()
    model = load_segmenter("unet", device)
    img = DocumentProcessor.pdf_page_to_image(pdf_path, page_index=page_idx, dpi=300)

    if img is None:
        print(f"[!] Failed to load {pdf_path}")
        return []

    techniques = build_techniques()
    rows = []
    doc_id = os.path.basename(pdf_path).replace(".pdf", "")

    print(f"\n{'='*80}\n[*] SWEEP: {doc_id} (page {page_idx})\n{'='*80}")

    for name, fn in techniques:
        print(f"[+] {name:35s}", end=" ", flush=True)

        prep = fn(img)
        probs = predict_layers(model, prep, device)

        for thr in THRESHOLDS:
            thr_dir = os.path.join(output_root, f"th_{thr}", doc_id)
            os.makedirs(thr_dir, exist_ok=True)

            mask_ns = (probs[0] > thr).astype(np.uint8) * 255
            mask_hw = (probs[1] > thr).astype(np.uint8) * 255
            mask_pr = (probs[2] > thr).astype(np.uint8) * 255

            Image.fromarray(mask_ns).save(os.path.join(thr_dir, f"{name}_ns.png"))
            Image.fromarray(mask_hw).save(os.path.join(thr_dir, f"{name}_hw.png"))
            Image.fromarray(mask_pr).save(os.path.join(thr_dir, f"{name}_pr.png"))

            overlay = img.astype(np.float32)
            for i in range(3):
                mask_i = probs[i] > thr
                alpha_i = mask_i[..., None].astype(np.float32) * 0.5
                overlay = (1 - alpha_i) * overlay + alpha_i * COLORS[i]

            Image.fromarray(overlay.astype(np.uint8)).save(
                os.path.join(thr_dir, f"{name}_composite.jpg"), quality=85)

            rows.append({
                "pdf_id": doc_id,
                "page_idx": page_idx,
                "technique": name,
                "threshold": thr,
                "noise_px": int((probs[0] > thr).sum()),
                "hw_px": int((probs[1] > thr).sum()),
                "printed_px": int((probs[2] > thr).sum()),
            })

        print("OK")

    return rows


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    output_root = os.path.join(BASE_PATH, "outputs", "debug", "preprocessing_sweep")
    os.makedirs(output_root, exist_ok=True)
    csv_path = os.path.join(output_root, "results.csv")

    pdfs = [(os.path.join(BASE_PATH, p), i) for p, i in TEST_PDFS if os.path.exists(os.path.join(BASE_PATH, p))]

    if not pdfs:
        print("[!] No test PDFs found.")
        sys.exit(1)

    num_techniques = len(build_techniques())

    print(f"\n{'='*80}")
    print("[*] UNet Preprocessing Sweep (CTR sweep 1.5/2.0/2.5 → BGD → GS)")
    print(f"    PDFs: {len(pdfs)}")
    print(f"    Techniques: {num_techniques}")
    print(f"    Thresholds: {THRESHOLDS}")
    print(f"    Total inferences: {len(pdfs) * num_techniques}")
    print(f"{'='*80}\n")

    all_rows = []
    with ProcessPoolExecutor(max_workers=min(len(pdfs), 4)) as exe:
        futs = {exe.submit(_run_pdf_sweep, p, i, output_root): (p, i)
                for p, i in pdfs}
        for fut in as_completed(futs):
            rows = fut.result()
            all_rows.extend(rows)
            pdf_path, page_idx = futs[fut]
            print(f"[✓] Done: {os.path.basename(pdf_path)}")

    with open(csv_path, "w", newline="") as f:
        fieldnames = ["pdf_id", "page_idx", "technique", "threshold", "noise_px", "hw_px", "printed_px"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n[*] Results saved to {csv_path}")
    print(f"    Rows: {len(all_rows)}")
