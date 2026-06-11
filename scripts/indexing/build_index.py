# -*- coding: utf-8 -*-
"""
Automated indexing pipeline for archival document processing

This module orchestrates the complete offline indexing workflow for the RADAR project. It initializes the segmentation, transcription (VLM), and noise classification models once and processes a curated subset of PDF documents. Each document page is rendered, analyzed for semantic regions and noise artifacts, and transcribed. The resulting structured records and derived visual artifacts are persisted to the filesystem for downstream search and analysis
"""

import os
import json
import argparse

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.gpu import DeviceManager
from src.preprocessing.pdf import DocumentProcessor
from src.segmentation.segformer import build_segformer
from src.segmentation.unet import build_unet
from src.segmentation.inference import load_segmenter
from src.detection.noise.predict import load_model as load_noise_model
from src.detection.vlm.qwen import QwenVLM
from src.indexing.pipeline import index_page
from src.indexing.stores import write_records_json

MODELS = {"segformer": build_segformer, "unet": build_unet}
PATH_DIR_OUTPUTS = os.path.join(BASE_PATH, "outputs")
PATH_DIR_RECORDS = os.path.join(PATH_DIR_OUTPUTS, "index", "records")
PATH_DIR_DERIVED = os.path.join(PATH_DIR_OUTPUTS, "derived")
PATH_SUBSET = os.path.join(PATH_DIR_OUTPUTS, "indexing", "subset.json")
MAX_PAGES = 5  # Cap the number of pages processed per document for indexing efficiency

def _stem(path):
    """
    Extract the filename without extension from a path
    """
    return os.path.splitext(os.path.basename(path))[0]

def index_pdf(pdf_path, models, device, dpi=300):
    """
    Process every page of a single PDF document

    Args:
        pdf_path (str): absolute path to the source PDF
        models (dict): collection of loaded models for segmentation, transcription, and noise
        device (torch.device): target hardware device for computation
        dpi (int): rasterization resolution for page rendering
    Returns:
        int: total count of records generated for the document
    """
    from pdf2image import pdfinfo_from_path

    doc_id = _stem(pdf_path)
    try:
        n_pages = pdfinfo_from_path(pdf_path).get("Pages", 1)
    except Exception:
        n_pages = 1
    print(f"[*] {doc_id}: {n_pages} pages")

    rec_dir = os.path.join(PATH_DIR_RECORDS, doc_id)
    total = 0
    for page_idx in range(min(n_pages, MAX_PAGES)):
        page_id = f"{doc_id}_p{page_idx:04d}"
        page_np = DocumentProcessor.pdf_page_to_image(pdf_path, page_index=page_idx, dpi=dpi)
        if page_np is None:
            print(f"[!] Failed to render {doc_id} page {page_idx}")
            continue

        records, _ = index_page(page_np, doc_id, page_id, pdf_path,
                                models["seg"], models["vlm"], models["noise"], device,
                                derived_dir=os.path.join(PATH_DIR_DERIVED, doc_id),
                                enhanced=True)
        write_records_json(records, rec_dir)  # Persist structured records to disk
        total += len(records)
        del page_np  # Free memory before next page processing
    return total

def main(args):
    """
    Execute the master indexing workflow
    """
    if not os.path.exists(PATH_SUBSET):
        raise FileNotFoundError(f"subset.json not found at {PATH_SUBSET}")

    with open(PATH_SUBSET, "r", encoding="utf-8") as f:
        subset = json.load(f)

    device = DeviceManager.get_device()
    DeviceManager.print_hardware_summary()

    models = {
        "seg": load_segmenter(args.model, device),
        "noise": load_noise_model(device=device)[0],
        "vlm": QwenVLM(device_map="auto"),
    }

    total = 0
    for epoch, paths in subset.items():
        print(f"[*] epoch={epoch} ({len(paths)} PDFs)")
        for pdf in sorted(paths):
            total += index_pdf(pdf, models, device)

    print(f"[*] Done - {total} records written under {PATH_DIR_RECORDS}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index PDFs from subset.json into structured records")
    parser.add_argument("--model", default="unet", choices=list(MODELS), help="Backbone model for segmentation")
    args = parser.parse_args()
    main(args)
