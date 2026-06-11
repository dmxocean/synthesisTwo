# -*- coding: utf-8 -*-
"""
General-purpose patch extraction and stitching utility for historical documents

Supports both the synthetic generation phase (with ground truth) and the inference phase (image-only)

Implements weighted-average stitching to eliminate edge artifacts in semantic masks
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class PatchRecord:
    patch_id: str
    row: int
    col: int
    offset_page: Tuple[int, int]
    kept: bool
    gt_layers: Dict[str, List[dict]] = field(default_factory=dict)
    file_paths: Dict[str, str] = field(default_factory=dict)

class DocumentSlicer:
    """
    Slice full-page images into overlapping patches for training or inference
    """
    def __init__(self, patch_size=512, stride=256, blank_threshold=0.03):
        self.sz = patch_size
        self.stride = stride
        self.blank = blank_threshold

    def slice_image(self, img_np: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int], str]]:
        """
        Slice a raw image into patches for sliding-window inference
        """
        H, W = img_np.shape[:2]
        pad = self.sz // 2
        img_pad = np.pad(img_np, ((pad, pad), (pad, pad), (0, 0)), mode="reflect") # Prevent edge artifacts during convolution
        H_pad, W_pad = img_pad.shape[:2]

        rows = range(0, H_pad - self.sz + 1, self.stride)
        cols = range(0, W_pad - self.sz + 1, self.stride)

        patches = []
        for ri, py in enumerate(rows):
            for ci, px in enumerate(cols):
                patch = img_pad[py:py + self.sz, px:px + self.sz]
                ox, oy = px - pad, py - pad # Back-calculate original page coordinates
                patch_id = f"r{ri:02d}_c{ci:02d}"
                patches.append((patch, (ox, oy), patch_id))
        return patches

    def stitch_masks(self, patch_masks: List[Tuple[np.ndarray, Tuple[int, int]]], target_size: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct a global mask from patches using weighted averaging
        """
        W, H = target_size
        C = patch_masks[0][0].shape[-1] if len(patch_masks[0][0].shape) > 2 else 1
        
        full_mask = np.zeros((H, W, C) if C > 1 else (H, W), dtype=np.float32)
        weight_mask = np.zeros((H, W), dtype=np.float32)

        # Weighted average calculation
        feather = np.ones((self.sz, self.sz), dtype=np.float32)
        ramp = np.linspace(0, 1, self.stride // 2)
        feather[:len(ramp), :] *= ramp[:, None] # Top edge
        feather[-len(ramp):, :] *= ramp[::-1, None] # Bottom edge
        feather[:, :len(ramp)] *= ramp[None, :] # Left edge
        feather[:, -len(ramp):] *= ramp[None, ::-1] # Right edge

        for mask, (ox, oy) in patch_masks:
            src_y0, src_x0 = max(0, -oy), max(0, -ox)
            src_y1, src_x1 = min(self.sz, H - oy), min(self.sz, W - ox)
            
            dst_y0, dst_x0 = max(0, oy), max(0, ox)
            dst_y1, dst_x1 = min(H, oy + self.sz), min(W, ox + self.sz)

            if dst_y1 > dst_y0 and dst_x1 > dst_x0:
                full_mask[dst_y0:dst_y1, dst_x0:dst_x1] += mask[src_y0:src_y1, src_x0:src_x1] * feather[src_y0:src_y1, src_x0:src_x1, None if C > 1 else Ellipsis]
                weight_mask[dst_y0:dst_y1, dst_x0:dst_x1] += feather[src_y0:src_y1, src_x0:src_x1]

        full_mask /= np.maximum(weight_mask[..., None] if C > 1 else weight_mask, 1e-8) # Avoid division by zero in unvisited pixels
        return full_mask

    def slice_with_gt(self, composite, masks, gt_layers, doc_id, output_dir):
        """
        Slice document and persist patches to disk with ground truth metadata
        
        Args:
            composite (np.ndarray): raw RGB image
            masks (Dict[str, np.ndarray]): semantic probability maps
            gt_layers (Dict[str, List[dict]]): bounding box annotations
            doc_id (str): unique identifier for the source document
            output_dir (str): absolute path to the patch storage directory
        Returns:
            List[PatchRecord]: metadata records for all extracted patches
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) # Ensure directory exists before writing patches

        H_orig, W_orig = composite.shape[:2]
        pad = self.sz // 2

        comp_pad = np.pad(composite, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        mask_pads = {k: np.pad(v, ((pad, pad), (pad, pad)), mode="reflect") for k, v in masks.items()}
        
        rows = range(0, comp_pad.shape[0] - self.sz + 1, self.stride)
        cols = range(0, comp_pad.shape[1] - self.sz + 1, self.stride)

        records = []
        for ri, py in enumerate(rows):
            for ci, px in enumerate(cols):
                patch_id = f"r{ri:02d}_c{ci:02d}"
                ox, oy = px - pad, py - pad
                comp_patch = comp_pad[py:py + self.sz, px:px + self.sz]
                
                if (comp_patch.mean(axis=2) < 220).sum() / comp_patch.size < self.blank: # Skip patches with insufficient foreground content
                    records.append(PatchRecord(patch_id, ri, ci, (ox, oy), False))
                    continue

                mask_patches = {k: v[py:py+self.sz, px:px+self.sz] for k, v in mask_pads.items()}
                gt_patch = self._clip_gt(gt_layers, ox, oy)
                
                # Persistence logic
                prefix = f"{doc_id}_{patch_id}"
                paths = {"composite": os.path.join(output_dir, f"{prefix}_composite.png")}
                Image.fromarray(comp_patch).save(paths["composite"])
                
                for k, v in mask_patches.items():
                    mpath = os.path.join(output_dir, f"{prefix}_mask_{k}.png")
                    Image.fromarray((v >= 10).astype(np.uint8)*255).save(mpath)
                    paths[f"mask_{k}"] = mpath

                records.append(PatchRecord(patch_id, ri, ci, (ox, oy), True, gt_patch, paths))

        return records

    def _clip_gt(self, gt_layers, ox, oy):
        """
        Translate and clip global annotations to local patch coordinates
        """
        clipped = {}
        for layer, anns in gt_layers.items():
            layer_anns = []
            for ann in anns:
                bbox = ann.get("bbox")
                if not bbox: continue
                
                x0, y0, x1, y1 = bbox
                cx0, cy0 = max(x0, ox), max(y0, oy)
                cx1, cy1 = min(x1, ox + self.sz), min(y1, oy + self.sz)
                
                if cx1 > cx0 and cy1 > cy0: # Keep only if there is a visible intersection
                    new_ann = dict(ann)
                    new_ann["bbox"] = [cx0 - ox, cy0 - oy, cx1 - ox, cy1 - oy]
                    layer_anns.append(new_ann)
            clipped[layer] = layer_anns
        return clipped
