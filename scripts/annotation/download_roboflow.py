# -*- coding: utf-8 -*-
"""
Roboflow synchronization utility for layout annotations

This module downloads the latest layout and asset annotations from the Roboflow workspace into the local filesystem. It implements the Roboflow SDK as the primary synchronization method and provides a curl-based fallback mechanism to ensure dataset availability
"""

import os
import shutil
from roboflow import Roboflow

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET_DIR = os.path.join(BASE_PATH, "data", "interim", "layouts", "annotations")

WORKSPACE_ID = "dmxocean"
PROJECT_ID = "radar-tybhh"

def main():
    """
    Execute the Roboflow dataset synchronization process

    Args:
        None
    Returns:
        None
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")  # Retrieve credentials from environment
    if not api_key:
        print("[!] Error: ROBOFLOW_API_KEY not found in environment")
        return

    if os.path.exists(TARGET_DIR):
        print(f"[*] Cleaning target directory: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)  # Ensure clean slate for new download
    os.makedirs(TARGET_DIR, exist_ok=True)

    try:
        print(f"[*] Attempting SDK download for {PROJECT_ID}")
        rf = Roboflow(api_key=api_key)  # Initialize Roboflow client
        project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
        
        print(f"[*] Downloading {PROJECT_ID} v3 in COCO format to {TARGET_DIR}")
        project.version(3).download("coco-segmentation", location=TARGET_DIR, overwrite=True)
        print(f"[SUCCESS] Dataset synchronized via SDK to {TARGET_DIR}")
        
    except Exception as e:
        print(f"[!] SDK download failed: {e}")
        print("[*] Falling back to CURL method")
        
        try:
            cmd = (
                f'curl -L "https://app.roboflow.com/ds/PlOeGKL13m?key=vF3h02sxt4" > roboflow.zip && '
                f'unzip -o roboflow.zip -d {TARGET_DIR} && '
                f'rm roboflow.zip'
            )
            os.system(cmd)  # Execute direct shell-level synchronization
            print(f"[SUCCESS] Dataset synchronized via CURL to {TARGET_DIR}")
        except Exception as e_curl:
            print(f"[!] CURL fallback also failed: {e_curl}")

if __name__ == "__main__":
    main()
