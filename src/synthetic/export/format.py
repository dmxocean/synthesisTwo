# -*- coding: utf-8 -*-
"""
Serialization utility for synthetic documents and metadata

This module implements the tiling and export logic for synthetic archival records, ensuring that high-resolution documents are correctly segmented into standard sizes while preserving spatial annotations and mask integrity
"""

import os
import json
import numpy as np
from PIL import Image

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration parameters
SIZE_TILE = 768
THRESHOLD_INK_PIXELS = 500  # Strictly skip tiles with low signal

class DataExporter:
    """
    Manages the persistence of synthetic data samples as tiles

    This class handles the export of generated documents into 768x768 segments, following the DELINE8K standard which includes RGB images and multi-frame TIFF labels. It also generates per-tile annotation JSON files for downstream COCO dataset construction
    """

    def __init__(self, path_output_root: str):
        """
        Initialize the exporter with a target directory structure
        """
        self.root = path_output_root
        self.path_dir_img = os.path.join(self.root, "images")
        self.path_dir_lbl = os.path.join(self.root, "labels")
        self.path_dir_ann = os.path.join(self.root, "annotations")

        # Directory initialization
        for d in [self.path_dir_img, self.path_dir_lbl, self.path_dir_ann]:
            os.makedirs(d, exist_ok=True)

    def _get_intersection_mask(self, binary_masks: list) -> np.ndarray:
        """
        Calculate boolean intersection where multiple classes are present
        """
        m_ns, m_hw, m_pr = binary_masks
        inter = (m_ns & m_hw) | (m_ns & m_pr) | (m_hw & m_pr)
        return (inter.astype(np.uint8) * 255)

    def _tile_annotations(self, annotations: list, left: int, top: int) -> list:
        """
        Filter and offset page-level annotations for a specific tile

        Args:
            annotations (list): Collection of page-level annotation dictionaries
            left (int): Horizontal offset of the tile in page coordinates
            top (int): Vertical offset of the tile in page coordinates
        Returns:
            list: Localized annotation dictionaries for the tile
        """
        right = left + SIZE_TILE
        bottom = top + SIZE_TILE
        result = []

        for ann in annotations:
            bx, by, bw, bh = ann["bbox"]
            
            # Intersection calculation
            ix = max(bx, left)
            iy = max(by, top)
            ix2 = min(bx + bw, right)
            iy2 = min(by + bh, bottom)
            
            if ix2 <= ix or iy2 <= iy:
                continue  # Annotation does not touch this tile

            entry = {
                "bbox": [ix - left, iy - top, ix2 - ix, iy2 - iy],
                "category": ann["category"],
            }

            gt_poly = ann.get("gt_polygon")
            if gt_poly:
                tile_poly = []
                for part in gt_poly:
                    adjusted = []
                    for i in range(0, len(part) - 1, 2):
                        adjusted.append(part[i] - left)
                        adjusted.append(part[i + 1] - top)
                    tile_poly.append(adjusted)
                entry["gt_polygon"] = tile_poly

            if "subtype" in ann:
                entry["subtype"] = ann["subtype"]

            result.append(entry)

        return result

    def save(self, name: str, image: Image.Image, layers: dict, masks: list, annotations: list) -> int:
        """
        Tile the document and persist valid 768x768 samples

        Args:
            name (str): Base name for the generated files
            image (Image.Image): High-resolution composite document image
            layers (dict): Dictionary of individual component layers
            masks (list): Unused parameter for future compatibility
            annotations (list): Full list of page-level annotations
        Returns:
            int: Number of valid tiles successfully saved to disk
        """
        w, h = image.size
        cols = w // SIZE_TILE
        rows = h // SIZE_TILE

        saved_tiles = 0
        for r in range(rows):
            for c in range(cols):
                left = c * SIZE_TILE
                top = r * SIZE_TILE
                right = left + SIZE_TILE
                bottom = top + SIZE_TILE
                box = (left, top, right, bottom)

                # Mask generation sequence
                mask_frames = []
                binary_masks = []

                for key in ("ns", "hw", "pr"):
                    layer_tile = layers[key].crop(box)
                    alpha = np.array(layer_tile)[..., 3]
                    mask_frames.append(Image.fromarray(alpha, mode="L"))
                    binary_masks.append(alpha > 0)

                total_ink = sum(m.sum() for m in binary_masks)
                if total_ink < THRESHOLD_INK_PIXELS:
                    continue  # Skip tiles with insufficient content

                inter_arr = self._get_intersection_mask(binary_masks)
                mask_frames.append(Image.fromarray(inter_arr, mode="L"))

                tile_name = f"{name}_t{saved_tiles:02d}"

                # Image persistence
                img_tile = image.crop(box)
                img_tile.save(os.path.join(self.path_dir_img, f"{tile_name}_input.png"))

                # Multi-frame label persistence
                mask_frames[0].save(
                    os.path.join(self.path_dir_lbl, f"{tile_name}_label.tiff"),
                    save_all=True,
                    append_images=mask_frames[1:],
                    compression="tiff_deflate"
                )

                # Metadata serialization
                tile_annos = self._tile_annotations(annotations, left, top)
                anno_data = {
                    "image": f"{tile_name}_input.png",
                    "width": SIZE_TILE,
                    "height": SIZE_TILE,
                    "annotations": tile_annos,
                }
                with open(os.path.join(self.path_dir_ann, f"{tile_name}.json"), "w", encoding="utf-8") as f:
                    json.dump(anno_data, f)

                saved_tiles += 1

        return saved_tiles
