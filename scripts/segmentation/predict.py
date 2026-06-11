# -*- coding: utf-8 -*-
"""
Segmentation prediction executable workflow

This script handles the prediction logic for segmentation models by loading trained checkpoints and processing input files. It supports various input formats including single images, directories, globs, and PDFs. Predictions are generated for each page, producing probability maps, confidence scores, and mask overlays
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
from pdf2image import pdfinfo_from_path

from src.core.config import get_artifact_dir
from src.segmentation.segformer import build_segformer
from src.segmentation.unet import build_unet
from src.segmentation.inference import predict_one
from src.preprocessing.pdf import DocumentProcessor
from src.core.gpu import DeviceManager

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Two independent segmentation models, shared inference for a clean A/B
MODELS = {"segformer": build_segformer, "unet": build_unet}


def default_checkpoint(model_name):
    """
    Construct the default checkpoint path for a given model
    """
    return os.path.join(get_artifact_dir("segmentation", model_name, "weights"), "best.pt")


def load_model(model_name, checkpoint, device):
    """
    Build the chosen model and load a checkpoint from the filesystem
    """
    model = MODELS[model_name](pretrained=False).to(device)
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint {checkpoint} (epoch {ckpt.get('epoch', '?')}, mIoU {ckpt.get('best_iou_mean', float('nan')):.4f})", flush=True)  # Log successful load
    else:
        print(f"Checkpoint not found at {checkpoint}. Running with untrained weights", flush=True)  # Log missing checkpoint
    model.eval()
    return model


def collect_inputs(pattern):
    """
    Expand a path or glob pattern into a list of image and PDF files
    """
    files = []
    for item in glob.glob(pattern):
        if os.path.isdir(item):
            for ext in ("png", "jpg", "pdf"):
                files.extend(glob.glob(os.path.join(item, "**", f"*.{ext}"), recursive=True))
        elif item.lower().endswith((".png", ".jpg", ".pdf")):
            files.append(item)
    return sorted(set(files))


def main(args):
    """
    Execution logic for the segmentation prediction pipeline
    """
    device = DeviceManager.get_device()  # Select primary compute device
    DeviceManager.print_hardware_summary()
    checkpoint = args.checkpoint or default_checkpoint(args.model)
    out_dir = args.output or get_artifact_dir("segmentation", args.model, "predictions")
    model = load_model(args.model, checkpoint, device)

    files = collect_inputs(args.input)
    print(f"Found {len(files)} files to process | model={args.model}", flush=True)  # Log total files

    for path in files:
        if path.lower().endswith(".pdf"):
            try:
                pages = pdfinfo_from_path(path).get("Pages", 1)
                stem_base = os.path.splitext(os.path.basename(path))[0]
                for p in range(pages):
                    img = DocumentProcessor.pdf_page_to_image(path, page_index=p)
                    if img is not None:
                        predict_one(model, img, f"{stem_base}_p{p:03d}", device, out_dir)
            except Exception as e:
                print(f"PDF failure {path}: {e}", flush=True)  # Log PDF processing error
        else:
            stem = os.path.splitext(os.path.basename(path))[0]
            img = np.array(Image.open(path).convert("RGB"))
            predict_one(model, img, stem, device, out_dir)

    print(f"Done - results in {out_dir}", flush=True)  # Log completion status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation sliding-window inference (segformer | unet)")
    parser.add_argument("--model", default="unet", choices=list(MODELS))
    parser.add_argument("--checkpoint", default="", help="defaults to outputs/segmentation/{model}/weights/best.pt")
    parser.add_argument("--input", required=True,
                        help="single PNG/PDF path, directory, or glob (searched recursively)")
    parser.add_argument("--output", default="", help="defaults to outputs/segmentation/{model}/predictions")
    args = parser.parse_args()
    main(args)
