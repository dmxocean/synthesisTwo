# -*- coding: utf-8 -*-
"""
Verification script for VLM inference and optimization logic

This script validates that the Qwen3-VL model is correctly integrated into the indexing pipeline and that the optimization logic properly filters redundant calls for known noise markers. It processes a sample archival page and compares the actual VLM call count against the expected number of unknown regions
"""

import os
import sys
import time
import torch
from PIL import Image

# Routes
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)

from src.core.gpu import DeviceManager
from src.preprocessing.pdf import DocumentProcessor
from src.segmentation.inference import load_segmenter
from src.detection.noise.predict import load_model as load_noise_model
from src.detection.vlm.qwen import QwenVLM
from src.indexing.pipeline import index_page
from src.core.confidence import LABEL_UNKNOWN

def run_smoke_test():
    """
    Execute the VLM optimization verification workflow
    Args:
        None
    Returns:
        None
    """
    device = DeviceManager.get_device()
    DeviceManager.print_hardware_summary()

    print("[*] Loading models")  # Status update for model initialization
    seg_model = load_segmenter("unet", device)
    noise_model = load_noise_model(device=device)[0]
    vlm = QwenVLM(device_map="auto")

    pdf_path = os.path.join(BASE_PATH, "data/raw/1938/guiradbcn_a1938m1.pdf")
    page_idx = 0
    doc_id = "smoke_test"
    page_id = "smoke_p0"

    print(f"[*] Processing {pdf_path} (page {page_idx})")  # Log source file details
    page_np = DocumentProcessor.pdf_page_to_image(pdf_path, page_index=page_idx, dpi=300)
    
    # Mock VLM describe_noise to track usage
    original_describe = vlm.describe_noise
    call_count = 0
    
    def counted_describe(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_describe(*args, **kwargs)
        
    vlm.describe_noise = counted_describe  # Inject counter into the VLM instance

    t0 = time.time()
    records, probs = index_page(
        page_np, doc_id, page_id, pdf_path,
        seg_model, vlm, noise_model, device,
        enhanced=True
    )
    elapsed = time.time() - t0

    # Result verification logic
    page_record = next(r for r in records if r.unit_type == "page")
    ns_marks = page_record.visual_marks
    unknown_count = sum(1 for m in ns_marks if m.mark_type == LABEL_UNKNOWN)
    described_count = sum(1 for m in ns_marks if m.description)

    print("VLM OPTIMIZATION SMOKE TEST RESULTS")
    print(f"Total processing time:     {elapsed:.2f}s")  # Target performance baseline is ~41s
    print(f"Total Noise regions:       {len(ns_marks)}")
    print(f"Unknown Noise regions:     {unknown_count}")
    print(f"VLM Description calls:     {call_count}")
    print(f"Described Noise records:   {described_count}")

    if call_count == unknown_count:
        print("[✓] SUCCESS: VLM was only called for unknown instances")
    else:
        print(f"[!] FAILURE: VLM call count ({call_count}) does not match unknown count ({unknown_count})")

    if elapsed < 60:
        print("[✓] SUCCESS: Processing time is within acceptable limits")
    else:
        print(f"[!] WARNING: Processing time ({elapsed:.2f}s) is higher than target")

if __name__ == "__main__":
    run_smoke_test()
