# -*- coding: utf-8 -*-
"""
Harvests IAM Handwriting assets and generates precise geometric metadata

This module replaces the legacy build.py script, providing a refined extraction process for IAM handwriting samples. It performs ink isolation using adaptive thresholding and morphological operations, captures hierarchical metadata from XML sources, and supports parallel processing for efficiency
"""

import os
import json
import argparse
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.metadata.harvester_core import relpath_posix

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIR_IAM_LIB = os.path.join(BASE_PATH, "data", "iam", "library")
PATH_DIR_RAW_XML = os.path.join(BASE_PATH, "data", "iam", "raw", "xml")
PATH_DIR_RAW_LINES = os.path.join(BASE_PATH, "data", "iam", "raw", "lines")
PATH_OUT_METADATA = os.path.join(BASE_PATH, "data", "metadata", "iam_handwriting.json")

THRESHOLD_C = 5  # Relaxed threshold to preserve stroke density


def clean_ink(img_np: np.ndarray) -> Image:
    """
    Isolates ink from a grayscale image and returns an RGBA mask

    Uses aggressive Gamma + Black-Hat + Biased Otsu to strip IAM paper
    """
    if img_np is None or img_np.size == 0:
        return None
    
    gamma = 0.8  # Contrast Stretching (Gamma Correction)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    darkened = cv2.LUT(img_np, table)

    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))  # Morphological Black-Hat
    blackhat = cv2.morphologyEx(darkened, cv2.MORPH_BLACKHAT, kernel_bh)

    if blackhat.size < 10:  # Fallback if too small for Otsu
        return None
        
    T, _ = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Aggressive Biased Otsu
    T_aggressive = min(255, T + 15)
    _, binary_mask = cv2.threshold(blackhat, T_aggressive, 255, cv2.THRESH_BINARY)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Morphological cleanup
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)  # Filter connected components
    filtered_mask = np.zeros_like(clean_mask)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 10 and w < (h * 30) and h < (w * 30):  # Keep blobs that aren't tiny speckles or absurdly long streaks
            filtered_mask[labels == i] = 255
            
    if np.sum(filtered_mask) == 0:
        return None
    
    h_out, w_out = img_np.shape  # Build RGBA output
    rgba = np.zeros((h_out, w_out, 4), dtype=np.uint8)
    rgba[:, :, 0:3] = img_np[:, :, np.newaxis]
    
    alpha_channel = cv2.bitwise_and(255 - img_np, filtered_mask)  # Optional: preserve natural texture in the alpha using grayscale intensity
    rgba[:, :, 3] = alpha_channel
    
    return Image.fromarray(rgba)


def parse_xml_metadata() -> List[Dict]:
    """
    Parses all IAM XML files and returns structured hierarchy
    """
    all_forms = []
    xml_files = sorted([f for f in os.listdir(PATH_DIR_RAW_XML) if f.endswith(".xml")])
    for fname in tqdm(xml_files, desc="Parsing IAM XML"):
        try:
            tree = ET.parse(os.path.join(PATH_DIR_RAW_XML, fname))
            root = tree.getroot()
            form_id = root.attrib["id"]
            writer_id = root.attrib["writer-id"]
            prefix = form_id.split("-")[0]
            lines = []
            for line_el in root.findall(".//line"):
                line_id = line_el.attrib["id"]
                line_text = line_el.attrib.get("text", "").replace("|", " ")
                cmps = line_el.findall(".//cmp")
                if not cmps:
                    continue
                l_x1 = min(int(c.attrib["x"]) for c in cmps)
                l_y1 = min(int(c.attrib["y"]) for c in cmps)
                l_x2 = max(int(c.attrib["x"]) + int(c.attrib["width"]) for c in cmps)
                l_y2 = max(int(c.attrib["y"]) + int(c.attrib["height"]) for c in cmps)
                line_bbox = [l_x1, l_y1, l_x2 - l_x1, l_y2 - l_y1]
                words = []
                for word_el in line_el.findall(".//word"):
                    w_cmps = word_el.findall(".//cmp")
                    if not w_cmps:
                        continue
                    w_x1 = min(int(c.attrib["x"]) for c in w_cmps)
                    w_y1 = min(int(c.attrib["y"]) for c in w_cmps)
                    w_x2 = max(int(c.attrib["x"]) + int(c.attrib["width"]) for c in w_cmps)
                    w_y2 = max(int(c.attrib["y"]) + int(c.attrib["height"]) for c in w_cmps)
                    words.append({"id": word_el.attrib["id"], "text": word_el.attrib.get("text", ""), "bbox": [w_x1, w_y1, w_x2 - w_x1, w_y2 - w_y1]})
                lines.append({"id": line_id, "text": line_text, "bbox": line_bbox, "words": words})
            all_forms.append({"id": form_id, "writer_id": writer_id, "prefix": prefix, "lines": lines})
        except Exception:
            continue
    return all_forms


def process_iam_task(task):
    """
    Worker function to process a single IAM asset (sentence or word)
    """
    t_type, asset_id, img_gs, text, save_dir, path_root = task
    
    mask = clean_ink(img_gs)
    if mask is None:
        return None
        
    os.makedirs(save_dir, exist_ok=True)
    png_fname = f"{asset_id}.png"
    png_path = os.path.join(save_dir, png_fname)
    mask.save(png_path)
    
    with open(os.path.join(save_dir, f"{asset_id}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
        
    return {
        "asset_id": asset_id,
        "path_rel": relpath_posix(png_path, path_root),
        "type": t_type,
        "text": text,
        "dims": {"w": mask.width, "h": mask.height}
    }


def run(workers=4):
    """
    Execute the IAM harvest pipeline in parallel
    """
    forms = parse_xml_metadata()
    if not forms:
        return
    print(f"[*] Planning IAM harvest with {workers} workers")
    
    tasks = []
    for form in forms:
        prefix, form_id = form["prefix"], form["id"]
        form_dir = os.path.join(PATH_DIR_RAW_LINES, prefix, form_id)
        if not os.path.isdir(form_dir):
            continue
            
        for line in form["lines"]:
            line_id = line["id"]
            src_path = os.path.join(form_dir, f"{line_id}.png")
            if not os.path.exists(src_path):
                continue
            line_img_gs = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if line_img_gs is None:
                continue
            
            s_save_dir = os.path.join(PATH_DIR_IAM_LIB, prefix, "sentences")  # Sentence Task
            tasks.append(("sentence", line_id, line_img_gs, line["text"], s_save_dir, BASE_PATH))
            
            lx, ly = line["bbox"][0], line["bbox"][1]  # Word Tasks
            for word in line["words"]:
                wx, wy, ww, wh = word["bbox"]
                rx, ry = max(0, wx - lx), max(0, wy - ly)
                rw, rh = min(ww, line_img_gs.shape[1] - rx), min(wh, line_img_gs.shape[0] - ry)
                if rw <= 2 or rh <= 2:
                    continue
                word_crop = line_img_gs[ry : ry + rh, rx : rx + rw].copy()
                w_save_dir = os.path.join(PATH_DIR_IAM_LIB, prefix, "words")
                tasks.append(("word", word["id"], word_crop, word["text"], w_save_dir, BASE_PATH))

    metadata = []
    print(f"[*] Harvesting {len(tasks)} assets")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_iam_task, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing IAM"):
            res = future.result()
            if res:
                metadata.append(res)

    os.makedirs(os.path.dirname(PATH_OUT_METADATA), exist_ok=True)
    with open(PATH_OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[*] IAM metadata saved to: {PATH_OUT_METADATA}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel IAM Harvester")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    run(workers=args.workers)
