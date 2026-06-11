# -*- coding: utf-8 -*-
"""
Standardizes DELINE8K dataset into the project format

Extracts the raw DELINE8K tarball, copies images, and processes multi-frame
TIFF labels into the project-standard 4-frame format (ns, hw, pr, inter)
"""

import os
import tarfile
import numpy as np
from PIL import Image
import shutil

# Routes
PATH_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAR_PATH     = os.path.join(PATH_ROOT, "DELINE8K.tar")
OUTPUT_ROOT  = os.path.join(PATH_ROOT, "data", "synthetic", "deline8k")
TEMP_EXTRACT = os.path.join(PATH_ROOT, "temp_d8k_extract")


def process_deline8k():
    """Extracts and processes the DELINE8K dataset
    
    Converts 3-channel labels to 4-channel project standard by recalculating
    the intersection mask
    """
    if not os.path.exists(TAR_PATH):
        print(f"[!] Error: {TAR_PATH} not found")
        return

    os.makedirs(os.path.join(OUTPUT_ROOT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "labels"), exist_ok=True)

    print(f"[*] Extracting {TAR_PATH}...")
    if os.path.exists(TEMP_EXTRACT):
        shutil.rmtree(TEMP_EXTRACT)    # Remove existing temp directory
    
    with tarfile.open(TAR_PATH) as tar:
        tar.extractall(path=TEMP_EXTRACT)

    img_dir = os.path.join(TEMP_EXTRACT, "images")
    lbl_dir = os.path.join(TEMP_EXTRACT, "labels")
    
    if not os.path.exists(img_dir):
        content = os.listdir(TEMP_EXTRACT)    # Handle potential nested folder inside tar
        if content:
            inner = content[0]
            candidate_img = os.path.join(TEMP_EXTRACT, inner, "images")
            if os.path.exists(candidate_img):
                img_dir = candidate_img
                lbl_dir = os.path.join(TEMP_EXTRACT, inner, "labels")

    if not os.path.exists(img_dir):
        print(f"[!] Error: Could not find images/ directory in {TEMP_EXTRACT}")
        return

    files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    total = len(files)
    print(f"[*] Processing {total} files...")

    for i, fname in enumerate(files):
        stem = fname.replace("_input.png", "").replace(".png", "")
        
        new_img_name = f"d8k_{stem}_input.png"
        shutil.copy2(os.path.join(img_dir, fname), 
                     os.path.join(OUTPUT_ROOT, "images", new_img_name))    # Copy image
        
        tiff_path = os.path.join(lbl_dir, f"{stem}_label.tiff")    # Process TIFF label
        if not os.path.exists(tiff_path):
            continue
            
        try:
            with Image.open(tiff_path) as img:
                # Mapping: 0->ns, 1->hw, 3->pr, 4->inter
                raw_frames = []
                for old_idx in [0, 1, 3]:    # NS, HW, PR
                    img.seek(old_idx)
                    raw_frames.append(np.array(img.convert("L")).copy())
                
                m_ns, m_hw, m_pr = [(f > 0) for f in raw_frames]    # Recalculate intersection
                inter = (m_ns & m_hw) | (m_ns & m_pr) | (m_hw & m_pr)
                inter_frame = (inter.astype(np.uint8) * 255)
                
                final_frames = [Image.fromarray(f, mode="L") for f in raw_frames]    # Assembly
                final_frames.append(Image.fromarray(inter_frame, mode="L"))
                
                new_lbl_name = f"d8k_{stem}_label.tiff"    # Save 4-frame TIFF
                final_frames[0].save(
                    os.path.join(OUTPUT_ROOT, "labels", new_lbl_name),
                    save_all=True,
                    append_images=final_frames[1:],
                    compression="tiff_deflate"
                )
        except Exception as e:
            print(f"[!] Error processing {tiff_path}: {e}")
            continue

        if (i + 1) % 500 == 0:
            print(f"  > Done {i+1}/{total}")

    shutil.rmtree(TEMP_EXTRACT)    # Cleanup
    print(f"[*] Success. Standardized data at {OUTPUT_ROOT}")


if __name__ == "__main__":
    process_deline8k()
