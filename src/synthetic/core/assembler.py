# -*- coding: utf-8 -*-
"""
Orchestrator for synthetic document assembly using geometry-aware polygon filling

This module composites various assets onto authentic paper backgrounds by utilizing precise polygon filling. It manages layers for noise, handwriting, and printed text, ensuring that content-tight polygons are derived for ground truth accuracy
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw
from src.core.config import PATH_DIR_PAPER
from src.core.geometry import GeometryKernel
from src.core.alpha import (
    content_mask_from_layer,
    polygon_from_content_tight,
    polygon_from_content_blobs,
)

PAPER_COLOR = (245, 242, 235)  # Fallback paper color
NOISE_SUBTYPES = frozenset({
    "circles_region",
    "lines_region",
    "crosses_region",
    "marks_region",
    "stamps_region",
})

def _rasterise_polygon(segmentation, width, height):
    """Convert polygon coordinates into a binary mask"""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for part in segmentation:
        pts = [(int(part[i]), int(part[i + 1])) for i in range(0, len(part) - 1, 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return mask

def _region_to_bbox(region):
    """Extract a bounding box from region metadata or polygon coordinates"""
    if "bbox" in region:
        return region["bbox"]
    raw = region.get("gt_polygon") or region.get("polygon")
    seg = raw[0] if isinstance(raw[0], list) else raw
    if isinstance(seg[0], (list, tuple)):
        xs, ys = [p[0] for p in seg], [p[1] for p in seg]
    else:
        xs, ys = seg[0::2], seg[1::2]
    x, y = int(min(xs)), int(min(ys))
    w, h = int(max(xs)) - x, int(max(ys)) - y
    return [x, y, max(1, w), max(1, h)]

def _ensure_polygon(region):
    """
    Ensure the presence of a polygon for a given region

    The function returns the existing ground truth polygon or derives a rectangular one from the bounding box if unavailable
    """
    if "gt_polygon" in region and region["gt_polygon"]:
        return region["gt_polygon"]
    bx, by, bw, bh = _region_to_bbox(region)
    return [[bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]]  # COCO-style polygon

class DocumentAssembler:
    """
    Manager for the composition of document regions onto paper backgrounds

    This class coordinates asset generation and placement, ensuring that each region is correctly rendered and its ground truth mask is refined based on actual content
    """

    def __init__(self, hw_gen, pr_gen, ns_gen):
        """Initialize the assembler with specific generators for each content type"""
        self.hw_gen = hw_gen
        self.pr_gen = pr_gen
        self.ns_gen = ns_gen

    def assemble(self, layout, mode="hybrid"):
        """
        Render a complete synthetic document based on a layout specification

        Args:
            layout (dict): Structural definition of the document regions
            mode (str): Generation mode for handwriting and printed text
        Returns:
            tuple: Composite image, layer dict, mask dict, and annotation list
        """
        w, h = layout["width"], layout["height"]
        epoch = layout.get("epoch", "monarchy")
        paper = self._load_paper(epoch, w, h)

        layers = {
            "hw": Image.new("RGBA", (w, h), (0, 0, 0, 0)),
            "pr": Image.new("RGBA", (w, h), (0, 0, 0, 0)),
            "ns": Image.new("RGBA", (w, h), (0, 0, 0, 0)),
        }
        masks = {
            "hw": Image.new("L", (w, h), 0),
            "pr": Image.new("L", (w, h), 0),
            "ns": Image.new("L", (w, h), 0),
        }

        all_annotations = []

        for region in layout["regions"]:
            bx, by, bw, bh = _region_to_bbox(region)
            label = region["type"]
            gt_poly = _ensure_polygon(region)

            mask_key = self._mask_key(label)
            if mask_key:
                poly_img = _rasterise_polygon(gt_poly, w, h)
                masks[mask_key].paste(Image.new("L", (w, h), 255), mask=poly_img)

            word_annos = []
            temp_layer = None
            if mask_key:
                temp_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))

            if label == "handwritten_region":
                annos = self.hw_gen.fill_polygon(temp_layer, gt_poly, epoch=epoch, mode=mode)
                word_annos.extend(annos)
            elif label == "printed_region":
                annos = self.pr_gen.fill_polygon(temp_layer, gt_poly, epoch=epoch, mode=mode)
                word_annos.extend(annos)
            elif label in NOISE_SUBTYPES:
                subtype = label.replace("_region", "")
                annos = self.ns_gen.fill_polygon(temp_layer, gt_poly, subtype=subtype, epoch=epoch)
                word_annos.extend(annos)

            if mask_key and not word_annos:
                print(f"[!] Empty fill: {label} at bbox=[{bx},{by},{bw},{bh}]", flush=True)  # Log missing content

            refined_poly = None
            if temp_layer is not None and gt_poly:
                content_mask = content_mask_from_layer(temp_layer, gt_poly, w, h)
                if label in NOISE_SUBTYPES:
                    refined_poly = polygon_from_content_tight(content_mask)  # Pixel-tight mask
                else:
                    refined_poly = polygon_from_content_blobs(content_mask)  # Line-level multi-polygon

            if temp_layer is not None:
                layers[mask_key].paste(temp_layer, (0, 0), temp_layer)
                paper.paste(layers[mask_key], (0, 0), layers[mask_key])

            anno = {
                "bbox": [bx, by, bw, bh],
                "category": self._category_name(label),
                "words": word_annos,
            }
            if gt_poly:
                if refined_poly:
                    anno["gt_polygon"] = refined_poly  # Tight to actual content
                    anno["template_polygon"] = gt_poly  # Preserved for traceability
                    anno["refined"] = True
                else:
                    anno["gt_polygon"] = gt_poly  # Use template as fallback
                    anno["refined"] = False
            if label in NOISE_SUBTYPES:
                anno["subtype"] = label.replace("_region", "")
            all_annotations.append(anno)

        return paper, layers, masks, all_annotations

    def _load_paper(self, epoch, w, h):
        """Select and resize a paper texture for the specified epoch"""
        epoch_paper_dir = os.path.join(PATH_DIR_PAPER, epoch)
        if os.path.exists(epoch_paper_dir):
            textures = [f for f in os.listdir(epoch_paper_dir) if f.endswith(".png")]
            if textures:
                chosen = random.choice(textures)
                tex = Image.open(os.path.join(epoch_paper_dir, chosen)).convert("RGB")
                return tex.resize((w, h), Image.Resampling.LANCZOS)
        return Image.new("RGB", (w, h), PAPER_COLOR)

    @staticmethod
    def _mask_key(label):
        """Map region labels to internal layer keys"""
        if label == "handwritten_region":
            return "hw"
        if label == "printed_region":
            return "pr"
        if label in NOISE_SUBTYPES:
            return "ns"
        return None

    @staticmethod
    def _category_name(label):
        """Normalize internal labels to human-readable category names"""
        if label == "handwritten_region":
            return "handwritten"
        if label == "printed_region":
            return "printed"
        return "noise"
