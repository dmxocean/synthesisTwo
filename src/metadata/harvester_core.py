# -*- coding: utf-8 -*-
"""
Core utilities for the Asset Harvesting & Intelligence Pipeline.
Handles COCO matching, epoch mapping, and polygon-based surgical cropping.
"""

import os
import re
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
from src.core.alpha import clean_real_asset_alpha
from src.core.config import DICT_EPOCHS

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH_COCO = os.path.join(BASE_PATH, "data", "interim", "layouts", "annotations", "train", "_annotations.coco.json")
PATH_TRAIN_IMAGES = os.path.join(BASE_PATH, "data", "interim", "layouts", "annotations", "train")


def relpath_posix(path, start):
    """
    Return a relative path using forward slashes on any OS
    """
    return os.path.relpath(path, start).replace(os.sep, "/")

class HarvesterCore:
    """
    Core utilities for the Asset Harvesting and Intelligence Pipeline
    """
    @staticmethod
    def get_epoch(filename):
        """
        Map a filename year to an epoch string
        """
        match = re.search(r"_a(\d{4})", filename)
        if not match: return "[NA]"
        year = int(match.group(1))
        for epoch, year_range in DICT_EPOCHS.items():
            if year in year_range: return epoch
        return "[NA]"

    @staticmethod
    def load_coco():
        """
        Load the Roboflow COCO database
        """
        if not os.path.exists(PATH_COCO):
            print(f"[!] COCO file not found: {PATH_COCO}")
            return None
        with open(PATH_COCO, "r") as f:
            return json.load(f)

    @staticmethod
    def surgical_crop(image_path, segmentation_poly):
        """
        Crop an asset using its polygon as an alpha mask to remove background

        Args:
            image_path (str): path to the source image
            segmentation_poly (list): polygon coordinates for masking
        Returns:
            Image: PIL RGBA image with asset isolated
        """
        if not isinstance(segmentation_poly, list) or not segmentation_poly:
            return None

        img = Image.open(image_path).convert("RGBA")

        all_pts = []
        for part in segmentation_poly:
            if not isinstance(part, list): continue
            pts = [(int(part[i]), int(part[i+1])) for i in range(0, len(part)-1, 2)]
            all_pts.extend(pts)

        if not all_pts: return None

        x_coords = [p[0] for p in all_pts]
        y_coords = [p[1] for p in all_pts]
        min_x, min_y = min(x_coords), min(y_coords)
        max_x, max_y = max(x_coords), max(y_coords)

        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        for part in segmentation_poly:
            if not isinstance(part, list): continue
            pts = [(int(part[i]), int(part[i+1])) for i in range(0, len(part)-1, 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)

        r, g, b, _ = img.split()
        img_with_mask = Image.merge("RGBA", (r, g, b, mask))
        cropped = img_with_mask.crop((min_x, min_y, max_x, max_y))
        return clean_real_asset_alpha(cropped)

    @staticmethod
    def analyze_noise_geometry(polygon):
        """
        Calculate high-precision geometric attributes for noise assets
        """
        if not isinstance(polygon, list) or not polygon:
            return {"width_oriented": 0, "height_oriented": 0, "angle": 0, "solidity": 0}

        pts = []
        for part in polygon:
            if not isinstance(part, list): continue
            pts.extend([(int(part[i]), int(part[i+1])) for i in range(0, len(part)-1, 2)])

        if not pts:
            return {"width_oriented": 0, "height_oriented": 0, "angle": 0, "solidity": 0}

        pts_np = np.array(pts, dtype=np.int32)

        rect = cv2.minAreaRect(pts_np)
        (cx, cy), (rw, rh), angle = rect

        if rw < rh:
            angle = angle - 90
            rw, rh = rh, rw

        area = cv2.contourArea(pts_np)
        hull = cv2.convexHull(pts_np)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        return {
            "width_oriented": round(float(rw), 2),
            "height_oriented": round(float(rh), 2),
            "angle": round(float(angle), 2),
            "solidity": round(float(solidity), 3)
        }