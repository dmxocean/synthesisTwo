# -*- coding: utf-8 -*-
"""
Segmentation inference components for document processing

This module implements per-page sliding-window inference for SegFormer and U-Net backbones. It manages the generation of probability maps, confidence scores, and layered masks for noise, handwriting, and printed text
"""

import os
import numpy as np
from PIL import Image
import torch

from src.core.slicer import DocumentSlicer
from src.core.confidence import uncertainty_map
from src.core.reporting import save_heatmap
from src.segmentation.data import NORM_MEAN, NORM_STD, KEY_LAYERS, COUNT_CLASSES
from src.preprocessing.normalize import normalise_background

SIZE_TILE = 768  # Tile size for sliding window
STRIDE_TILE = 384  # Stride between tiles

COLORS = {
    0: np.array([0, 150, 255]),  # Noise (blue)
    1: np.array([255, 50, 50]),  # Handwriting (red)
    2: np.array([50, 255, 50]),  # Printed (green)
}

def _normalise(img_np, mean=NORM_MEAN, std=NORM_STD):
    """
    Apply ImageNet normalization to an image array

    Args:
        img_np (np.ndarray): Input image in uint8 format (H, W, 3)
        mean (tuple): Normalization mean
        std (tuple): Normalization standard deviation
    Returns:
        np.ndarray: Normalized float32 tensor (3, H, W)
    """
    arr = img_np.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))

def load_segmenter(model_name, device, ckpt_root=None):
    """
    Initialize a segmenter and load its best checkpoint

    Args:
        model_name (str): Identifier for the model architecture ('segformer' or 'unet')
        device (torch.device): Target computation device
        ckpt_root (str, optional): Root directory for checkpoints
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    from src.segmentation.segformer import build_segformer
    from src.segmentation.unet import build_unet
    from src.core.config import get_artifact_dir
    builders = {"segformer": build_segformer, "unet": build_unet}

    model = builders[model_name](pretrained=False).to(device)
    ckpt_root = ckpt_root or get_artifact_dir("segmentation", model_name, "weights")
    ckpt_path = os.path.join(ckpt_root, "best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"[*] Segmenter {model_name}: loaded {ckpt_path}", flush=True)  # Report successful load
    else:
        print(f"[!] Segmenter checkpoint not found at {ckpt_path} (untrained weights)", flush=True)  # Warn on missing weights
    model.eval()
    return model

def predict_layers(model, image_np, device):
    """
    Execute feathered sliding-window inference across an image

    Args:
        model (torch.nn.Module): Trained segmentation model
        image_np (np.ndarray): Background-normalized RGB page (H, W, 3)
        device (torch.device): Computation device
    Returns:
        np.ndarray: Multi-label probability map (C, H, W) in range [0, 1]
    """
    slicer = DocumentSlicer(patch_size=SIZE_TILE, stride=STRIDE_TILE)
    patches = slicer.slice_image(image_np)
    H, W = image_np.shape[:2]

    patch_probs = []
    model.eval()
    with torch.no_grad():
        for patch, (ox, oy), _ in patches:
            tile_t = torch.from_numpy(_normalise(patch)).unsqueeze(0).to(device)
            with torch.amp.autocast(device.type):
                logits = model(tile_t)
            prob_chw = torch.sigmoid(logits).float().cpu().numpy()[0]  # Extract probabilities
            patch_probs.append((prob_chw.transpose(1, 2, 0), (ox, oy)))  # Channel-last for stitching

    stitched = slicer.stitch_masks(patch_probs, target_size=(W, H))
    return stitched.transpose(2, 0, 1)  # Restore channel-first format

def save_layer_artifacts(probs, image_np, ink_mask, stem, out_dir,
                         save_confidence=True, use_ink_gate=True):
    """
    Persist per-page segmentation artifacts to disk

    This method generates raw probability files, confidence maps, binary layer masks, and visualization overlays
    Args:
        probs (np.ndarray): Probability map (C, H, W)
        image_np (np.ndarray): Original RGB image
        ink_mask (np.ndarray): Binary ink mask
        stem (str): Filename stem for artifacts
        out_dir (str): Output directory path
        save_confidence (bool): Whether to export confidence/uncertainty maps
        use_ink_gate (bool): Whether to mask predictions with the ink mask
    Returns:
        dict: Pixel counts for each predicted layer
    """
    from src.core.config import SEG_THRESHOLDS
    os.makedirs(out_dir, exist_ok=True)

    thrs = np.array(SEG_THRESHOLDS, dtype=np.float32)[:, None, None]
    if use_ink_gate:
        pred_binary = (probs > thrs) & ink_mask  # Mask with ink detection
    else:
        pred_binary = (probs > thrs)  # Use raw thresholds only

    np.save(os.path.join(out_dir, f"{stem}.prob.npy"), probs.astype(np.float32))  # Save raw probabilities

    if save_confidence:
        conf = probs.max(axis=0).astype(np.float32)
        np.save(os.path.join(out_dir, f"{stem}.conf.npy"), conf)  # Save max probability map
        save_heatmap(uncertainty_map(probs), os.path.join(out_dir, f"{stem}_conf.png"),
                     title=f"{stem} - uncertainty (1 - max prob)")  # Export uncertainty visualization

    for idx, key in enumerate(KEY_LAYERS):
        Image.fromarray((pred_binary[idx].astype(np.uint8) * 255), mode="L").save(
            os.path.join(out_dir, f"{stem}_{key}.png"))  # Save individual layer masks

    overlay = image_np.astype(np.float32)
    for class_id in range(COUNT_CLASSES):
        alpha = pred_binary[class_id][..., None].astype(np.float32) * 0.5
        overlay = (1 - alpha) * overlay + alpha * COLORS[class_id]  # Apply color-coded overlays
    Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(
        os.path.join(out_dir, f"{stem}_overlay.png"))

    counts = {key: int(pred_binary[idx].sum()) for idx, key in enumerate(KEY_LAYERS)}
    print(f"[*] {stem}: " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)  # Log pixel distribution
    return counts

def predict_one(model, image_np, stem, device, path_out_dir, save_confidence=True):
    """
    Process a single document page through the entire inference pipeline

    Args:
        model (torch.nn.Module): Trained segmentation model
        image_np (np.ndarray): Original RGB page (H, W, 3)
        stem (str): Output filename identifier
        device (torch.device): Computation device
        path_out_dir (str): Target output directory
        save_confidence (bool): Whether to generate confidence maps
    Returns:
        dict: Cumulative pixel counts for the page layers
    """
    image_norm, ink_mask = normalise_background(image_np)
    probs = predict_layers(model, image_norm, device)
    return save_layer_artifacts(probs, image_np, ink_mask, stem, path_out_dir, save_confidence)
