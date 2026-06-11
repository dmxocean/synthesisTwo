# -*- coding: utf-8 -*-
"""
High-precision noise generator with strict-subtype semantics

This module implements the logic for rendering realistic or procedural noise artifacts within document regions. It prioritizes real assets from the manual_noise collection while falling back to subtype-appropriate procedural shapes when suitable assets are unavailable
"""

import os
import random
from PIL import Image, ImageDraw
from src.core.geometry import GeometryKernel
from src.core.alpha import load_manual_asset

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration parameters
COLOR_INK = (40, 40, 40, 220)  # Near-black with mild translucency
COLOR_LINE = (60, 60, 60, 230)  # Slightly lighter grey for lines

class NoiseGenerator:
    """
    Polygonal region filler for noise artifacts

    This class manages the placement of stamps, circles, crosses, and other noise types within specified bounds. It ensures that the generated noise maintains the subtype identity defined in the layout templates
    """

    def __init__(self, metadata_provider):
        """
        Initialize the generator with a metadata source
        """
        self.metadata = metadata_provider

    def fill_polygon(self, canvas: Image.Image, polygon: list, subtype: str = None, epoch: str = None) -> list:
        """
        Render a real noise asset or fall back to a procedural shape

        Args:
            canvas (Image.Image): RGBA layer for in-place rendering
            polygon (list): COCO segmentation format coordinates
            subtype (str): Specific noise category like stamps or crosses
            epoch (str): Optional epoch tag for asset filtering
        Returns:
            list: Annotation data for the placed noise instance
        """
        bounds = GeometryKernel.get_polygon_bounds(polygon)
        bw, bh = bounds[2] - bounds[0], bounds[3] - bounds[1]

        asset_meta = self.metadata.query_assets("manual_noise", bw, bh, subtype=subtype, epoch=epoch)

        if asset_meta:
            asset_path = os.path.join(os.getcwd(), asset_meta["path_rel"])
            asset_img = load_manual_asset(asset_path)  # Refine alpha to remove paper bleed-through
            if asset_img is not None:
                cx = bounds[0] + bw // 2
                cy = bounds[1] + bh // 2
                x = cx - asset_img.width // 2
                y = cy - asset_img.height // 2

                canvas.paste(asset_img, (x, y), asset_img)
                return [{
                    "text": "",
                    "subtype": asset_meta.get("subtype", subtype or "noise"),
                    "bbox": [x, y, asset_img.width, asset_img.height]
                }]

        return [self._draw_procedural(canvas, bounds, subtype)]

    def generate_block(self, width: int, height: int, page_y_ratio: float = 0.5, epoch: str = None, subtype: str = None) -> tuple:
        """
        Create a standalone RGBA block populated with one noise instance
        """
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        poly = [[0, 0, width, 0, width, height, 0, height]]
        return canvas, self.fill_polygon(canvas, poly, subtype=subtype, epoch=epoch)

    @staticmethod
    def _draw_procedural(canvas: Image.Image, bounds: list, subtype: str) -> dict:
        """
        Render a subtype-specific procedural shape inside the polygon bounds

        This method generates geometric approximations for noise categories when no suitable manual assets are discovered. It adjusts dimensions to fit the requested polygon bounding box
        """
        bx, by = bounds[0], bounds[1]
        bw = max(1, bounds[2] - bounds[0])
        bh = max(1, bounds[3] - bounds[1])
        cx = bx + bw // 2
        cy = by + bh // 2
        draw = ImageDraw.Draw(canvas)

        if subtype == "stamps":
            pad = max(4, int(min(bw, bh) * 0.05))
            thickness = max(3, int(min(bw, bh) * 0.04))
            draw.ellipse([bx + pad, by + pad, bx + bw - pad, by + bh - pad],
                         outline=COLOR_INK, width=thickness)
            
            # Double-ring stamp silhouette
            inner_pad = pad + thickness * 2
            if bw - 2 * inner_pad > 10 and bh - 2 * inner_pad > 10:
                draw.ellipse([bx + inner_pad, by + inner_pad, bx + bw - inner_pad, by + bh - inner_pad],
                             outline=COLOR_INK, width=max(2, thickness // 2))

        elif subtype == "circles":
            pad = max(4, int(min(bw, bh) * 0.10))
            thickness = max(2, int(min(bw, bh) * 0.03))
            draw.ellipse([bx + pad, by + pad, bx + bw - pad, by + bh - pad],
                         outline=COLOR_INK, width=thickness)

        elif subtype == "crosses":
            pad = max(4, int(min(bw, bh) * 0.10))
            thickness = max(3, int(min(bw, bh) * 0.05))
            draw.line([bx + pad, by + pad, bx + bw - pad, by + bh - pad],
                      fill=COLOR_INK, width=thickness)
            draw.line([bx + pad, by + bh - pad, bx + bw - pad, by + pad],
                      fill=COLOR_INK, width=thickness)

        elif subtype == "lines":
            thickness = max(2, int(bh * 0.35))  # Cover majority of height for line representation
            pad = max(4, int(bw * 0.02))
            draw.line([bx + pad, cy, bx + bw - pad, cy],
                      fill=COLOR_LINE, width=thickness)

        elif subtype == "marks":
            thickness = max(2, int(min(bw, bh) * 0.04))
            steps = max(3, bw // 30)
            amp = max(2, bh // 4)
            pts = []
            for i in range(steps + 1):
                x = bx + int(bw * i / steps)
                y = cy + ((-1) ** i) * amp
                pts.append((x, y))
            draw.line(pts, fill=COLOR_INK, width=thickness)

        return {"text": "", "subtype": subtype or "noise", "bbox": [bx, by, bw, bh]}
