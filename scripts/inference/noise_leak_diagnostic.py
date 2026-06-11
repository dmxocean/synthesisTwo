# -*- coding: utf-8 -*-
"""
Diagnostic tool for noise layer leak analysis

This script evaluates the impact of different probability thresholds on noise region extraction. It helps identify 'leakage' where valid text might be incorrectly classified as noise by visualizing the resulting bounding boxes at various sensitivity levels
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_PATH)

from src.preprocessing.pdf import DocumentProcessor
from src.preprocessing.normalize import preprocess_ctr_bgd_gs
from src.segmentation.inference import load_segmenter, predict_layers
from src.detection.noise.bbox import extract_noise_bboxes
from src.core.gpu import DeviceManager


def run_diagnostic(pdf_path, page_idx, model, device, output_root):
    """
    Execute noise extraction diagnostic for a specific page across multiple thresholds

    Args:
        pdf_path (str): Path to the source PDF file
        page_idx (int): Zero-based index of the page to process
        model (torch.nn.Module): Loaded segmentation model
        device (torch.device): Target compute device
        output_root (str): Root directory for diagnostic visualizations
    Returns:
        None
    """
    doc_name = os.path.basename(pdf_path).replace(".pdf", "")
    page_id = f"{doc_name}_p{page_idx}"
    out_dir = os.path.join(output_root, "noise_diagnostic", page_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] DIAGNOSTIC: {page_id}")

    img = DocumentProcessor.pdf_page_to_image(pdf_path, page_index=page_idx, dpi=300)
    if img is None:
        return

    inp, _ = preprocess_ctr_bgd_gs(img)
    probs = predict_layers(model, inp, device)
    prob_ns = probs[0]  # Noise channel

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"{'Threshold':<12} | {'NS Regions':<12}")
    print("-" * 27)

    for th in thresholds:
        binary = (prob_ns > th).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for c in contours:
            if cv2.contourArea(c) < 200:  # Minimum blob area filter
                continue
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append([x, y, w, h])
        
        count = len(bboxes)
        print(f"{th:<12.1f} | {count:<12}")

        canvas = img.copy()  # Visualization generation
        for x, y, w, h in bboxes:
            cv2.rectangle(canvas, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        Image.fromarray(canvas).save(os.path.join(out_dir, f"th_{th:.1f}.jpg"), quality=80)

    print(f"[*] Visualizations saved to {out_dir}")


if __name__ == "__main__":
    device = DeviceManager.get_device()
    model = load_segmenter("unet", device)

    pdf = os.path.join(BASE_PATH, "data", "raw", "1938", "guiradbcn_a1938m1.pdf")
    if os.path.exists(pdf):
        run_diagnostic(pdf, 0, model, device, os.path.join(BASE_PATH, "outputs", "debug"))
    else:
        print(f"[!] Path not found: {pdf}")
