# -*- coding: utf-8 -*-
"""
High-precision handwriting generator with multi-source selection

This module implements the core logic for filling polygonal regions with synthetic or real handwritten assets. It supports multiple generation modes, including realistic, synthetic, and hybrid, ensuring visual coherence at the region level while allowing for diverse document-level composition
"""

import os
import random
from PIL import Image
from src.core.geometry import GeometryKernel
from src.core.alpha import load_manual_asset

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration parameters
MANUAL_RATIO = 0.5  # Probability of manual_htr as primary source in hybrid mode
GAP_LINE_VERTICAL = (15, 35)  # Natural line spacing in real manuscripts
GAP_WORD_HORIZONTAL = (25, 60)  # High variability for human-written appearance
WIDTH_THRESHOLD_SENTENCE = 950  # Sentence asset threshold for remaining width
TARGET_HEIGHT_SENTENCE = 250  # Height hint for full-line strips
TARGET_HEIGHT_WORD = 150  # Height hint for narrow row tails
MIN_HEIGHT_REQUIRED = 15  # Minimum polygon height for rendering
MIN_WIDTH_FOR_PLACEMENT = 10  # Minimum width to attempt placement
MIN_WIDTH_TAIL_FILL = 50  # Trailing gap threshold for tail fill
EMPTY_ROW_STEP = 5  # Small probe step for empty intervals

class HandwritingGenerator:
    """
    Polygonal region filler for handwritten content

    This class manages the dense placement of handwritten snippets within arbitrary polygonal bounds. It coordinates between local manual assets and the IAM dataset to achieve the requested visual style and density
    """

    def __init__(self, metadata_provider):
        """
        Initialize the generator with a metadata source
        """
        self.metadata = metadata_provider

    def fill_polygon(self, canvas: Image.Image, polygon: list, epoch: str = None, mode: str = "hybrid") -> list:
        """
        Densely fill a polygon with source-selected handwriting snippets

        Args:
            canvas (Image.Image): RGBA layer for in-place rendering
            polygon (list): COCO segmentation format coordinates
            epoch (str): Optional epoch tag for asset filtering
            mode (str): Generation mode among realistic, synthetic, or hybrid
        Returns:
            list: Collection of per-word annotations with labels and bounding boxes
        """
        # Source selection logic
        if mode == "realistic":
            use_manual_first, allow_fallback = True, False
        elif mode == "synthetic":
            use_manual_first, allow_fallback = False, False
        else:
            use_manual_first = random.random() < MANUAL_RATIO
            allow_fallback = True

        primary = self._place_manual if use_manual_first else self._place_iam
        secondary = self._place_iam if use_manual_first else self._place_manual

        bounds = GeometryKernel.get_polygon_bounds(polygon)
        y_min, y_max = bounds[1], bounds[3]

        curr_y = y_min
        all_bboxes = []

        while curr_y < y_max - MIN_HEIGHT_REQUIRED:
            segments = GeometryKernel.get_polygon_intervals(polygon, curr_y)
            if not segments:
                curr_y += EMPTY_ROW_STEP  # Small probe for thin polygons
                continue

            max_h_this_row = 40
            row_had_placement = False

            for x_start, x_end in segments:
                curr_x = x_start
                while curr_x < x_end - MIN_WIDTH_FOR_PLACEMENT:
                    width_remaining = x_end - curr_x

                    placed = primary(canvas, curr_x, curr_y, width_remaining, epoch)
                    if placed is None and allow_fallback:
                        placed = secondary(canvas, curr_x, curr_y, width_remaining, epoch)

                    if placed is None:
                        break  # Abandon row tail upon placement failure

                    asset_w, asset_h, text_label = placed
                    all_bboxes.append({
                        "text": text_label,
                        "bbox": [curr_x, curr_y, asset_w, asset_h]
                    })
                    max_h_this_row = max(max_h_this_row, asset_h)
                    row_had_placement = True
                    curr_x += asset_w + random.randint(*GAP_WORD_HORIZONTAL)

                # Secondary pass for tail filling
                if allow_fallback:
                    gap = x_end - curr_x
                    if gap > MIN_WIDTH_TAIL_FILL:
                        placed = secondary(canvas, curr_x, curr_y, gap, epoch)
                        if placed is not None:
                            asset_w, asset_h, text_label = placed
                            all_bboxes.append({
                                "text": text_label,
                                "bbox": [curr_x, curr_y, asset_w, asset_h]
                            })
                            max_h_this_row = max(max_h_this_row, asset_h)
                            row_had_placement = True

            # Vertical advancement logic
            if row_had_placement:
                curr_y += max_h_this_row + random.randint(*GAP_LINE_VERTICAL)
            else:
                curr_y += EMPTY_ROW_STEP

        return all_bboxes

    def generate_block(self, width: int, height: int, writer_id: str = None) -> tuple:
        """
        Create an isolated RGBA block on a rectangular canvas
        """
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        poly = [[0, 0, width, 0, width, height, 0, height]]
        bboxes = self.fill_polygon(canvas, poly)
        return canvas, bboxes

    def _place_manual(self, canvas: Image.Image, curr_x: int, curr_y: int, width_avail: int, epoch: str) -> tuple:
        """
        Retrieve and render a snippet from the manual_htr collection
        """
        asset_meta = self.metadata.query_assets(
            "manual_htr", width_avail, TARGET_HEIGHT_WORD, epoch=epoch
        )
        if not asset_meta:
            return None
        
        loaded = self._load_asset(asset_meta, is_manual=True)
        if loaded is None:
            return None
            
        canvas.paste(loaded, (curr_x, curr_y), loaded)
        return loaded.width, loaded.height, asset_meta.get("text", "")

    def _place_iam(self, canvas: Image.Image, curr_x: int, curr_y: int, width_avail: int, epoch: str) -> tuple:
        """
        Retrieve and render a snippet from the IAM handwriting collection
        """
        target_h = TARGET_HEIGHT_SENTENCE if width_avail > WIDTH_THRESHOLD_SENTENCE else TARGET_HEIGHT_WORD
        asset_meta = self.metadata.query_assets(
            "iam_handwriting", width_avail, target_h, epoch=epoch
        )
        if not asset_meta:
            return None
            
        loaded = self._load_asset(asset_meta, is_manual=False)
        if loaded is None:
            return None
            
        canvas.paste(loaded, (curr_x, curr_y), loaded)
        return loaded.width, loaded.height, asset_meta.get("text", "")

    @staticmethod
    def _load_asset(asset_meta: dict, is_manual: bool) -> Image.Image:
        """
        Open and prepare an RGBA asset for rendering

        Args:
            asset_meta (dict): Metadata containing the relative path to the asset
            is_manual (bool): Flag indicating if the asset requires alpha refinement
        Returns:
            Image.Image: Processed RGBA image or None if unreadable
        """
        asset_path = os.path.join(os.getcwd(), asset_meta["path_rel"])
        if is_manual:
            return load_manual_asset(asset_path)
            
        try:
            return Image.open(asset_path).convert("RGBA")
        except Exception:
            return None  # Skip asset upon read failure
