# -*- coding: utf-8 -*-
"""
Canonical PDF-to-image conversion for the synthesis pipeline

Standardizes all source documents to 2480 x 3508 px (A4 at 300 DPI) regardless of input dimensions, ensuring consistent tensor shapes across all downstream models
"""

import os
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIZE_A4 = (2480, 3508)  # Width x height in pixels at 300 DPI
RATIO_A4 = 2480 / 3508  # Portrait aspect ratio

class DocumentProcessor:
    """
    Normalization of historical document scans into canonical tensor shapes

    This class provides static methods to convert PDF pages and arbitrary images into standardized formats suitable for neural network input
    """

    @staticmethod
    def pdf_to_standard_image(pdf_path, page_index=0, dpi=300):
        """
        Convert a single PDF page to a standardized A4 RGB image

        Args:
            pdf_path (str): Absolute path to the source PDF
            page_index (int): Zero-based page index within the PDF
            dpi (int): Rasterization resolution
        Returns:
            np.ndarray: Normalized RGB image or None if conversion fails
        """
        try:
            pages = convert_from_path(pdf_path, dpi=dpi,
                                      first_page=page_index + 1,
                                      last_page=page_index + 1)
            if not pages:
                return None
            return DocumentProcessor.standardize_image(np.array(pages[0].convert("RGB")))
        except Exception as e:
            print(f"[!] Render failure ({os.path.basename(pdf_path)} p{page_index}): {e}")  # Report conversion errors
            return None

    @staticmethod
    def standardize_image(img_np):
        """
        Resize image to canonical A4 dimensions while maintaining aspect ratio

        The process center-crops the input to match the A4 aspect ratio before performing a final resize to 2480 x 3508 pixels
        """
        h, w = img_np.shape[:2]
        current_ratio = w / h

        if current_ratio > RATIO_A4:
            target_w = int(h * RATIO_A4)
            start_x = (w - target_w) // 2
            cropped = img_np[:, start_x:start_x + target_w]  # Crop horizontally from center
        else:
            target_h = int(w / RATIO_A4)
            start_y = (h - target_h) // 2
            cropped = img_np[start_y:start_y + target_h, :]  # Crop vertically from center

        return cv2.resize(cropped, SIZE_A4, interpolation=cv2.INTER_LANCZOS4)  # High-quality Lanczos resampling

    @staticmethod
    def pdf_page_to_image(pdf_path, page_index=0, dpi=300):
        """
        Rasterize one PDF page at the given DPI

        The method preserves the page's native dimensions without cropping or resizing
        IMPORTANT: This does not force A4 dimensions and is intended for sliding-window inference
        Args:
            pdf_path (str): Absolute path to the source PDF
            page_index (int): Zero-based page index
            dpi (int): Rasterization resolution
        Returns:
            np.ndarray: RGB image at native page dimensions or None on failure
        """
        try:
            pages = convert_from_path(pdf_path, dpi=dpi,
                                      first_page=page_index + 1,
                                      last_page=page_index + 1)
            if not pages:
                return None
            return np.array(pages[0].convert("RGB"))
        except Exception as e:
            print(f"[!] Render failure ({os.path.basename(pdf_path)} p{page_index}): {e}")  # Report rendering errors
            return None

    @staticmethod
    def get_ink_mask(img_rgb, threshold=180):
        """
        Generate a binary mask isolating ink pixels

        The mask identifies pixels below the specified luminance threshold and inverts them to represent ink as positive values
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)  # Invert to make ink positive
        return mask
