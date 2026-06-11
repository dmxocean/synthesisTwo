# -*- coding: utf-8 -*-
"""
Roboflow upload utility for document layout samples

This module automates the synchronization of sampled document images to the Roboflow project. It handles batching by historical epoch, applies relevant metadata tags, and implements skip logic for already uploaded assets to prevent duplication
"""

import os
from roboflow import Roboflow

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLES_DIR = os.path.join(BASE_PATH, "data", "interim", "layouts", "samples")

WORKSPACE_ID = "dmxocean"
PROJECT_ID = "sonar"

def main():
    """
    Execute the image upload pipeline to Roboflow

    Args:
        None
    Returns:
        None
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")  # Retrieve credentials from environment
    if not api_key:
        print("[!] Error: ROBOFLOW_API_KEY not found in environment")
        return

    try:
        rf = Roboflow(api_key=api_key)  # Initialize Roboflow client
        project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
        print(f"[*] Connected to Roboflow: {WORKSPACE_ID}/{PROJECT_ID}")
    except Exception as e:
        print(f"[!] Initialization failed: {e}")
        return

    if not os.path.exists(SAMPLES_DIR):
        print(f"[!] Data root not found: {SAMPLES_DIR}")
        return

    print("[*] Fetching uploaded images from Roboflow")
    existing = project.search_all(fields=["name"])  # Query existing dataset manifest
    uploaded_names = {img["name"] for batch in existing for img in batch}
    print(f"[*] Already in project: {len(uploaded_names)} images")

    epochs = [d for d in os.listdir(SAMPLES_DIR) if os.path.isdir(os.path.join(SAMPLES_DIR, d))]
    print(f"[*] Epochs found: {', '.join(sorted(epochs)) if epochs else 'none'}")

    total_uploaded = 0
    total_skipped = 0
    total_failed = 0

    for epoch in sorted(epochs):
        epoch_dir = os.path.join(SAMPLES_DIR, epoch)
        images = [f for f in os.listdir(epoch_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        to_upload = [img for img in images if img not in uploaded_names]  # Filter out existing images
        already_done = len(images) - len(to_upload)
        if already_done:
            print(f"[~] {epoch.capitalize()}: skipping {already_done} already uploaded")
        if not to_upload:
            total_skipped += already_done
            continue

        display_name = epoch.capitalize()
        print(f"[*] Batch: {display_name} | {len(to_upload)} to upload")

        for i, img_name in enumerate(to_upload, 1):
            img_path = os.path.join(epoch_dir, img_name)

            try:
                project.upload(
                    image_path=img_path,
                    batch_name=display_name,
                    tag_names=[display_name],
                    split="train",
                    num_retry_uploads=3
                )
                uploaded_names.add(img_name)  # Update local cache of remote state
                total_uploaded += 1
                print(f"    [{i}/{len(to_upload)}] OK: {img_name}")
            except Exception as e:
                total_failed += 1
                print(f"    [{i}/{len(to_upload)}] FAIL: {img_name} -> {e}")

        total_skipped += already_done

    print(f"[*] Done - uploaded: {total_uploaded} | skipped: {total_skipped} | failed: {total_failed}")

if __name__ == "__main__":
    main()
