# -*- coding: utf-8 -*-
"""
Layout augmentation utilities for synthetic document generation

This module provides structural transformations applied to document templates before assembly. It implements a three-phase augmentation pipeline: class-preserving region pruning, cross-template region injection, and tile-zone gap filling to ensure sufficient visual density and diversity in the generated samples
"""

import math
import random
import os
import numpy as np
from collections import defaultdict

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration parameters
N_MAX_REGIONS = 25  # Phase 2 density cap
N_VARIANTS_DEFAULT = 20  # Default variants per template
N_VARIANTS_WARN = 100  # Threshold for scarcity warning
TILE_SIZE = 768  # Pixel size for export tiles
MIN_TILE_DENSITY = 0.40  # Minimum coverage fraction per tile
N_MAX_REGIONS_DENSE = 250  # Maximum additions for Phase 3

def _region_bbox(region: dict) -> list:
    """
    Calculate the bounding box for a region dictionary
    """
    if "bbox" in region:
        return region["bbox"]
    
    raw = region.get("gt_polygon") or region.get("polygon")
    if not raw or not isinstance(raw, list):
        return [0, 0, 1, 1]  # Degenerate fallback
    
    try:
        all_pts = []
        for part in raw:
            if not isinstance(part, list):
                continue
            all_pts.extend([(part[i], part[i+1]) for i in range(0, len(part)-1, 2)])
        
        if not all_pts:
            return [0, 0, 1, 1]
            
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs)) - x, int(max(ys)) - y
        return [x, y, max(1, w), max(1, h)]
    except Exception:
        return [0, 0, 1, 1]

def prune_regions(regions: list, prune_rate: float = 0.40) -> list:
    """
    Remove a portion of regions per class while maintaining class presence
    """
    by_class = defaultdict(list)
    for i, reg in enumerate(regions):
        by_class[reg["type"]].append(i)
        
    keep = set(range(len(regions)))
    for cls, indices in by_class.items():
        if len(indices) <= 1:
            continue  # Protect singleton classes
            
        max_removable = len(indices) - 1
        k = min(max_removable, round(len(indices) * prune_rate))
        if k > 0:
            to_remove = random.sample(indices, k)
            keep -= set(to_remove)
            
    return [regions[i] for i in sorted(keep)]

def build_occupancy(regions: list, page_w: int, page_h: int, gap: int = 20) -> np.ndarray:
    """
    Generate a boolean occupancy bitmap for the current regions
    """
    occ = np.zeros((page_h, page_w), dtype=bool)
    for reg in regions:
        bx, by, bw, bh = _region_bbox(reg)
        x1, y1 = max(0, bx - gap), max(0, by - gap)
        x2, y2 = min(page_w, bx + bw + gap), min(page_h, by + bh + gap)
        occ[y1:y2, x1:x2] = True
    return occ

def inject_regions(regions: list, epoch_pool: list, page_w: int, page_h: int, n_inject: int = 5, min_gap: int = 20) -> list:
    """
    Inject donor regions from the epoch pool into available white space
    """
    occ = build_occupancy(regions, page_w, page_h, gap=min_gap)
    result, accepted = list(regions), 0
    pool = list(epoch_pool)
    random.shuffle(pool)
    
    for donor in pool:
        if accepted >= n_inject or len(result) >= N_MAX_REGIONS:
            break
            
        bx, by, bw, bh = _region_bbox(donor)
        x1, y1 = max(0, bx - min_gap), max(0, by - min_gap)
        x2, y2 = min(page_w, bx + bw + min_gap), min(page_h, by + bh + min_gap)
        
        if occ[y1:y2, x1:x2].any():
            continue  # Avoid overlaps
            
        result.append(donor)
        occ[y1:y2, x1:x2] = True
        accepted += 1
        
    return result

def _tile_bbox_coverage(regions: list, tile_x: int, tile_y: int, tile_size: int = TILE_SIZE) -> float:
    """
    Calculate the bounding box coverage fraction for a specific tile zone
    """
    tile_area = tile_size * tile_size
    covered = 0
    for reg in regions:
        bx, by, bw, bh = _region_bbox(reg)
        ix1 = max(bx, tile_x)
        iy1 = max(by, tile_y)
        ix2 = min(bx + bw, tile_x + tile_size)
        iy2 = min(by + bh, tile_y + tile_size)
        if ix2 > ix1 and iy2 > iy1:
            covered += (ix2 - ix1) * (iy2 - iy1)
    return min(covered, tile_area) / tile_area

def _relocate_to_tile(donor: dict, tile_x: int, tile_y: int, tile_size: int = TILE_SIZE) -> dict:
    """
    Translate a donor region to a random position within a target tile
    """
    bx, by, bw, bh = _region_bbox(donor)
    if bw > tile_size or bh > tile_size:
        return None

    new_x = tile_x + random.randint(0, max(0, tile_size - bw))
    new_y = tile_y + random.randint(0, max(0, tile_size - bh))
    dx, dy = new_x - bx, new_y - by

    new_region = dict(donor)
    new_region["bbox"] = [new_x, new_y, bw, bh]

    raw = donor.get("gt_polygon")
    if raw:
        new_region["gt_polygon"] = [
            [v + (dx if j % 2 == 0 else dy) for j, v in enumerate(part)]
            for part in raw
        ]

    return new_region

def fill_tile_gaps(regions: list, epoch_pool: list, page_w: int, page_h: int, min_density: float = MIN_TILE_DENSITY, tile_size: int = TILE_SIZE, max_additions: int = N_MAX_REGIONS_DENSE) -> list:
    """
    Ensure every tile zone meets the minimum density requirement via injection
    """
    cols = page_w // tile_size
    rows = page_h // tile_size
    result = list(regions)
    n_added = 0

    candidates = [
        d for d in epoch_pool
        if _region_bbox(d)[2] <= tile_size and _region_bbox(d)[3] <= tile_size
    ]

    for tr in range(rows):
        for tc in range(cols):
            if n_added >= max_additions:
                return result

            tile_x = tc * tile_size
            tile_y = tr * tile_size

            if _tile_bbox_coverage(result, tile_x, tile_y, tile_size) >= min_density:
                continue

            pool = list(candidates)
            random.shuffle(pool)

            for donor in pool:
                if n_added >= max_additions:
                    break
                if _tile_bbox_coverage(result, tile_x, tile_y, tile_size) >= min_density:
                    break

                relocated = _relocate_to_tile(donor, tile_x, tile_y, tile_size)
                if relocated is not None:
                    result.append(relocated)
                    n_added += 1

    return result

def augment_layout(base_layout: dict, epoch_pool: list, n_inject: int = 5, prune_rate: float = 0.40, min_gap: int = 20) -> dict:
    """
    Apply the full augmentation pipeline to a base layout template
    """
    page_w, page_h = base_layout["width"], base_layout["height"]
    pruned = prune_regions(base_layout["regions"], prune_rate=prune_rate)
    mixed = inject_regions(pruned, epoch_pool, page_w, page_h,
                            n_inject=n_inject, min_gap=min_gap)
    mixed = fill_tile_gaps(mixed, epoch_pool, page_w, page_h)
    return {**base_layout, "regions": mixed}

def plan_variants(count: int, all_templates: list, target_eras: list) -> dict:
    """
    Calculate the number of variants needed per template across epochs

    Args:
        count (int): Total desired document count
        all_templates (list): Discovery list from template directory scanning
        target_eras (list): Epoch identifiers requested for synthesis
    Returns:
        dict: Variants-per-template mapping keyed by epoch
    """
    present = [e for e in target_eras if any(ep == e for ep, _ in all_templates)]
    if not present:
        return {}
        
    if count is None:
        return {e: N_VARIANTS_DEFAULT for e in present}
        
    per_epoch = math.ceil(count / len(present))
    return {
        e: max(1, math.ceil(per_epoch / sum(1 for ep, _ in all_templates if ep == e)))
        for e in present
    }

def warn_if_scarce(variants_by_epoch: dict, threshold: int = N_VARIANTS_WARN):
    """
    Display a warning when template counts are insufficient for requested volume
    """
    high = {e: v for e, v in variants_by_epoch.items() if v > threshold}
    if high:
        details = ", ".join(f"{e}={v}" for e, v in high.items())
        print(f"[!] Template scarcity: variants-per-template above {threshold} "
              f"({details}) - visual diversity bounded by template count",
              flush=True)
