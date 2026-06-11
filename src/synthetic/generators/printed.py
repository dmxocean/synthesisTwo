# -*- coding: utf-8 -*-
"""
High-precision printed text generator with multi-source selection

This module implements the core logic for filling polygonal regions with synthetic or real printed assets. It supports multiple generation modes, including realistic, synthetic, and hybrid, ensuring visual coherence at the region level while allowing for diverse document-level composition
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont
from src.core.geometry import GeometryKernel
from src.core.textgen import VOCAB_SPANISH
from src.core.alpha import load_manual_asset
from src.core.config import PATH_FILE_FONT

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration parameters
MANUAL_RATIO = 0.5  # Probability of manual_ocr as primary source in hybrid mode
GAP_LINE_VERTICAL = 12  # Vertical spacing between lines
GAP_WORD_HORIZONTAL = (20, 45)  # Horizontal spacing between words
SIZE_FONT_DEFAULT = 32  # Default font size for TTF rendering
MIN_HEIGHT_REQUIRED = 12  # Minimum polygon height for rendering
MIN_WIDTH_REQUIRED = 8  # Minimum width to attempt placement
EMPTY_ROW_STEP = 5  # Small probe step for empty intervals

class PrintedGenerator:
    """
    Polygonal region filler for printed content

    This class manages the dense placement of printed snippets within arbitrary polygonal bounds. It coordinates between local manual OCR assets and TrueType fonts to achieve the requested visual style and density
    """

    def __init__(self, metadata_provider):
        """
        Initialize the generator with a metadata source
        """
        self.metadata = metadata_provider
        self._font = None  # Lazily loaded per process worker

    def fill_polygon(self, canvas: Image.Image, polygon: list, epoch: str = None, mode: str = "hybrid") -> list:
        """
        Densely fill a polygon with source-selected printed snippets

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

        primary = self._place_manual if use_manual_first else self._place_font
        secondary = self._place_font if use_manual_first else self._place_manual

        bounds = GeometryKernel.get_polygon_bounds(polygon)
        y_min, y_max = bounds[1], bounds[3]

        curr_y = y_min
        all_bboxes = []

        while curr_y < y_max - MIN_HEIGHT_REQUIRED:
            segments = GeometryKernel.get_polygon_intervals(polygon, curr_y)
            if not segments:
                curr_y += EMPTY_ROW_STEP  # Small probe for thin polygons
                continue

            max_h = SIZE_FONT_DEFAULT
            row_had_placement = False
            for x_start, x_end in segments:
                curr_x = x_start
                while curr_x < x_end - MIN_WIDTH_REQUIRED:
                    width_avail = x_end - curr_x

                    placed = primary(canvas, curr_x, curr_y, width_avail, epoch)
                    if placed is None and allow_fallback:
                        placed = secondary(canvas, curr_x, curr_y, width_avail, epoch)

                    if placed is None:
                        break  # Abandon row tail upon placement failure

                    asset_w, asset_h, text_label = placed
                    all_bboxes.append({
                        "text": text_label,
                        "bbox": [curr_x, curr_y, asset_w, asset_h]
                    })
                    max_h = max(max_h, asset_h)
                    row_had_placement = True
                    curr_x += asset_w + random.randint(*GAP_WORD_HORIZONTAL)

            # Vertical advancement logic
            if row_had_placement:
                curr_y += max_h + GAP_LINE_VERTICAL
            else:
                curr_y += EMPTY_ROW_STEP

        return all_bboxes

    def generate_block(self, width: int, height: int, font_size: int = SIZE_FONT_DEFAULT) -> tuple:
        """
        Create an isolated RGBA block on a rectangular canvas
        """
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        poly = [[0, 0, width, 0, width, height, 0, height]]
        return canvas, self.fill_polygon(canvas, poly)

    def _place_manual(self, canvas: Image.Image, curr_x: int, curr_y: int, width_avail: int, epoch: str) -> tuple:
        """
        Retrieve and render a snippet from the manual_ocr collection
        """
        asset_meta = self.metadata.query_assets("manual_ocr", width_avail, 150, epoch=epoch)
        if not asset_meta:
            return None
            
        asset_path = os.path.join(os.getcwd(), asset_meta["path_rel"])
        asset_img = load_manual_asset(asset_path)  # Refine alpha to remove paper bleed-through
        if asset_img is None:
            return None
            
        canvas.paste(asset_img, (curr_x, curr_y), asset_img)
        return asset_img.width, asset_img.height, asset_meta.get("text", "[PRINTED]")

    def _place_font(self, canvas: Image.Image, curr_x: int, curr_y: int, width_avail: int, epoch: str = None) -> tuple:
        """
        Render a TrueType word and paste it onto the canvas
        """
        font = self._get_font()
        
        # Candidate selection
        candidates = random.sample(VOCAB_SPANISH, min(6, len(VOCAB_SPANISH)))
        for word in candidates:
            word_img = self._render_word(font, word)
            if word_img.width <= width_avail:
                canvas.paste(word_img, (curr_x, curr_y), word_img)
                return word_img.width, word_img.height, word
                
        # Shortest word fallback
        shortest = min(VOCAB_SPANISH, key=len)
        word_img = self._render_word(font, shortest)
        if word_img.width <= width_avail:
            canvas.paste(word_img, (curr_x, curr_y), word_img)
            return word_img.width, word_img.height, shortest
            
        return None

    def _get_font(self) -> ImageFont.FreeTypeFont:
        """
        Retrieve the cached TrueType font or load it from disk
        """
        if self._font is None:
            self._font = ImageFont.truetype(PATH_FILE_FONT, SIZE_FONT_DEFAULT)
        return self._font

    @staticmethod
    def _render_word(font: ImageFont.FreeTypeFont, word: str) -> Image.Image:
        """
        Rasterize a single word onto a transparent RGBA canvas
        """
        bbox = font.getbbox(word)
        w = max(1, bbox[2] - bbox[0] + 4)
        h = max(1, bbox[3] - bbox[1] + 4)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((2, 2 - bbox[1]), word, font=font, fill=(20, 20, 20, 230))
        return img
