# -*- coding: utf-8 -*-
"""
Smoke test suite for segmentation layer validation

This script processes a set of representative PDF pages through the segmentation model and generates visual overlays for each layer (noise, handwritten, printed). It serves as a quick verification tool to ensure model outputs are consistent and layers are correctly separated
"""

import os
import sys

import numpy as np
from PIL import Image

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_PATH)

from src.preprocessing.pdf import DocumentProcessor
from src.preprocessing.normalize import preprocess_ctr_bgd_gs
from src.segmentation.inference import load_segmenter, predict_layers, COLORS
from src.core.gpu import DeviceManager


def save_smoke_test(pdf_path, page_idx, model, device, output_root):
    """
    Execute a segmentation smoke test on a specific PDF page

    Args:
        pdf_path (str): Path to the source PDF file
        page_idx (int): Zero-based index of the page to process
        model (torch.nn.Module): Loaded segmentation model
        device (torch.device): Target compute device
        output_root (str): Root directory for smoke test results
    Returns:
        None
    """
    doc_name = os.path.basename(pdf_path).replace(".pdf", "")
    page_id = f"{doc_name}_p{page_idx}"
    out_dir = os.path.join(output_root, page_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Processing {page_id}")

    img = DocumentProcessor.pdf_page_to_image(pdf_path, page_index=page_idx, dpi=300)
    if img is None:
        return
    Image.fromarray(img).save(os.path.join(out_dir, "0_original.jpg"), quality=85)

    inp, ink_mask = preprocess_ctr_bgd_gs(img)
    Image.fromarray(inp).save(os.path.join(out_dir, "1_preprocessed.jpg"), quality=85)

    probs = predict_layers(model, inp, device)

    layer_names = ["noise", "handwritten", "printed"]
    for i, name in enumerate(layer_names):
        mask = (probs[i] > 0.2).astype(np.uint8) * 255
        Image.fromarray(mask).save(os.path.join(out_dir, f"2_mask_{name}.png"))
        print(f"    - {name}: {(probs[i] > 0.2).sum():,} pixels")

    overlay = img.astype(np.float32)
    for i in range(3):
        mask_binary = (probs[i] > 0.2)
        alpha = mask_binary[..., None].astype(np.float32) * 0.5
        overlay = (1 - alpha) * overlay + alpha * COLORS[i]

    Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(out_dir, "3_composite_overlay.jpg"), quality=85)
    print(f"[*] Results saved to {out_dir}")


if __name__ == "__main__":
    device = DeviceManager.get_device()
    model = load_segmenter("unet", device)

    output_root = os.path.join(BASE_PATH, "outputs", "debug", "layers_smoke_test")
    os.makedirs(output_root, exist_ok=True)

    test_pages = [
        ("data/raw/1925/guiradbcn_a1925m2.pdf", 0),
        ("data/raw/1938/guiradbcn_a1938m1.pdf", 0),
        ("data/raw/1938/guiradbcn_a1938m5.pdf", 0),
        ("data/raw/1945/guiradbcn_a1945m6d10.pdf", 0),
        ("data/raw/1950/guiradbcn_a1950m1d2.pdf", 0),
    ]

    for pdf_rel, idx in test_pages:
        pdf_abs = os.path.join(BASE_PATH, pdf_rel)
        if os.path.exists(pdf_abs):
            save_smoke_test(pdf_abs, idx, model, device, output_root)
        else:
            print(f"[!] Path not found: {pdf_abs}")
