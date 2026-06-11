# -*- coding: utf-8 -*-
"""
Mathematical utilities for complex polygon manipulation and scanline intersection

This module provides tools for calculating geometric properties of irregular shapes, specifically for milimetric precision in asset placement. It handles horizontal interval calculations for scanline filling, axis-aligned bounding box extraction, and aspect ratio computations for COCO-formatted segmentation polygons
"""

import os
from typing import List, Tuple

import numpy as np

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_PATH, "data")
STORAGE_DIR = os.path.join(BASE_PATH, "storage")

class GeometryKernel:
    """
    Core engine for geometry-aware regional filling and spatial analysis
    """

    @staticmethod
    def get_polygon_intervals(segmentation: List[List[float]], y: int) -> List[Tuple[int, int]]:
        """
        Calculate horizontal intervals of a polygon at a specific vertical coordinate
        """
        if not segmentation or not isinstance(segmentation, list):
            return []

        intersections = []
        for part in segmentation:
            if not isinstance(part, list):
                continue
            pts = [(int(part[i]), int(part[i + 1])) for i in range(0, len(part) - 1, 2)]
            if len(pts) < 3:
                continue
            
            for j in range(len(pts)):
                p1 = pts[j]
                p2 = pts[(j + 1) % len(pts)]
                
                if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                    t = (y - p1[1]) / (p2[1] - p1[1])
                    intersect_x = p1[0] + t * (p2[0] - p1[0])
                    intersections.append(intersect_x)
        
        intersections.sort()
        
        intervals = []
        for i in range(0, len(intersections) - 1, 2):
            x_start = int(intersections[i])
            x_end = int(intersections[i + 1])
            if x_end > x_start:
                intervals.append((x_start, x_end))
        
        return intervals

    @staticmethod
    def get_polygon_bounds(segmentation: List[List[float]]) -> List[int]:
        """
        Retrieve the axis-aligned bounding box for a given polygon
        """
        if not segmentation or not isinstance(segmentation, list):
            return [0, 0, 0, 0]

        all_x = []
        all_y = []
        for part in segmentation:
            if not isinstance(part, list):
                continue
            all_x.extend(part[0::2])
            all_y.extend(part[1::2])
        
        if not all_x:
            return [0, 0, 0, 0]
            
        return [int(min(all_x)), int(min(all_y)), int(max(all_x)), int(max(all_y))]

    @staticmethod
    def get_aspect_ratio(segmentation: List[List[float]]) -> float:
        """
        Calculate the width-to-height aspect ratio of a polygon boundary
        """
        bx = GeometryKernel.get_polygon_bounds(segmentation)
        w = bx[2] - bx[0]
        h = bx[3] - bx[1]
        return w / h if h > 0 else 1.0
