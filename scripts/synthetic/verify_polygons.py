# -*- coding: utf-8 -*-
"""
Visual verification harness for content-tight gt_polygon refinement

Runs a small synthetic batch through the modified pipeline and emits
side-by-side comparison images so a human reviewer can confirm that:

  1. Refined gt_polygon outlines hug the actual ink of stamps, crosses,
     and other noise instances instead of the loose Roboflow template region
  2. Refined polygons for handwritten and printed regions track the text
     envelope rather than the over-large templated rectangle
  3. Per-asset alpha cleaning successfully strips paper bleed-through from
     manually cropped real assets across varying inks and saturations

Inputs:  layout templates under data/layouts/templates/{epoch}/
Outputs: data/synthetic/verify/polygons/{mode}/
           - standard pipeline outputs (images, layers, masks, annotations)
           - visualizations/pages/  side-by-side template-vs-refined overlays
           - visualizations/assets/ per-asset original-vs-cleaned panels

Invocation: python -m src.synthetic.verify.polygons --count 3 --modes all --prewarm

Critical design decisions:
  Generation runs through the regular generate_sample worker so the renders
  are exactly what production will emit, the visualization step is then a
  pure read-only post-process driven by the annotation JSON files that
  carry both template_polygon and refined gt_polygon
"""

import os
import glob
import json
import math
import argparse
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont

from src.synthetic.factory import generate_sample
from src.synthetic.layouts.augment import plan_variants, warn_if_scarce
from src.synthetic.providers.metadata import MetadataProvider
import cv2
from src.core.alpha import SIZE_KERNEL_CLOSE
from src.core.config import EPOCHS as EPOCH_ALL

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIR_TEMPLATES = os.path.join(PATH_ROOT, "data", "layouts", "templates")
PATH_DIR_VERIFY_POLY = os.path.join(PATH_ROOT, "data", "synthetic", "verify", "polygons")
PATH_DIR_MANUAL_ROOT = os.path.join(PATH_ROOT, "data", "assets", "manual")

# Catalog names that source manually cropped assets requiring alpha cleaning
CATALOGS_MANUAL = ["manual_noise", "manual_htr", "manual_ocr"]

# Constants
MODES_ALL = ["realistic", "synthetic", "hybrid"]
BASE_SEED = 42

# Per-class template/refined colors so the reviewer can scan-spot which class
# is failing without cross-referencing the annotation JSON  Template colors sit
# on the warm/saturated side, refined colors on the cooler/lighter side
COLOR_TEMPLATE_BY_CAT = {
    "noise":       (220,  50,  50),  # Red
    "printed":     (230, 130,  50),  # Orange
    "handwritten": (200,  70, 200),  # Magenta
}
COLOR_REFINED_BY_CAT = {
    "noise":       ( 50, 200,  80),  # Green
    "printed":     ( 80, 220,  80),  # Lime
    "handwritten": ( 80, 200, 220),  # Cyan
}
COLOR_CONTENT_BY_KEY = {
    "ns": (230,  60, 200),          # Magenta tint for the noise content mask
    "pr": (255, 150,   0),          # Amber tint for printed content
    "hw": ( 80, 180, 240),          # Blue tint for handwritten content
}
COLOR_FALLBACK = (160, 160, 160)    # Used when an annotation lacks a recognised category
ALPHA_CONTENT  = 110                # Transparency for the per-region content-mask overlay panel
WIDTH_OUTLINE  = 5                  # Bumped from 3 for visibility against complex backgrounds
ALPHA_FILL     = 50                 # Translucent fill inside polygons


def _list_templates(templates_root, epoch_list):
    """
    Returns [(epoch, tpl_path)] for every .json template in the requested epoch dirs

    Args:
        templates_root (str):       absolute path to data/layouts/templates
        epoch_list     (list[str]): epoch names to walk

    Returns:
        list[tuple]: pairs of (epoch_name, template_json_path)
    """
    result = []
    for epoch in epoch_list:
        epoch_dir = os.path.join(templates_root, epoch)
        if not os.path.isdir(epoch_dir): continue  # Epoch directory absent or empty
        for fname in sorted(os.listdir(epoch_dir)):
            if fname.endswith(".json"):
                result.append((epoch, os.path.join(epoch_dir, fname)))
    return result


def _load_template_regions(path):
    """
    Reads a template JSON and returns its regions list

    Args:
        path (str): absolute path to a Roboflow template JSON

    Returns:
        list: regions array, empty when the key is missing
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("regions", [])


def _build_tasks(count_arg, epoch_list, modes):
    """
    Plans the synthetic generation task list across the requested modes

    Args:
        count_arg  (int):       number of layouts per mode
        epoch_list (list[str]): epochs to draw templates from
        modes      (list[str]): subset of MODES_ALL

    Returns:
        list[tuple]: argument tuples ready to feed generate_sample workers
    """
    all_templates = _list_templates(PATH_DIR_TEMPLATES, epoch_list)
    if not all_templates:
        print("[!] No templates found", flush=True)
        return []

    epoch_pools = {}
    for epoch in epoch_list:
        donor_regions = []
        for ep, tpl_path in all_templates:
            if ep == epoch:
                donor_regions.extend(_load_template_regions(tpl_path))
        epoch_pools[epoch] = donor_regions

    variants_by_epoch = plan_variants(count_arg, all_templates, epoch_list)
    warn_if_scarce(variants_by_epoch)
    per_epoch_target = (math.ceil(count_arg / len(variants_by_epoch))
                       if count_arg is not None and variants_by_epoch else None)

    tasks   = []
    doc_idx = 0
    for epoch, v in variants_by_epoch.items():
        epoch_templates = [(e, p) for e, p in all_templates if e == epoch]
        epoch_docs      = 0
        for t_idx, (_, tpl_path) in enumerate(epoch_templates):
            for k in range(v):
                if per_epoch_target is not None and epoch_docs >= per_epoch_target: break
                if count_arg is not None and doc_idx >= count_arg: break
                seed = BASE_SEED * 10000 + doc_idx
                for mode in modes:
                    mode_output_root = os.path.join(PATH_DIR_VERIFY_POLY, mode)
                    tasks.append((
                        epoch, tpl_path, k, t_idx, seed, mode_output_root,
                        epoch_pools[epoch], mode
                    ))
                doc_idx    += 1
                epoch_docs += 1
            if count_arg is not None and doc_idx >= count_arg: break
    return tasks


def _draw_polygon_overlay(image, polygon, color, width):
    """
    Draws a translucent polygon outline with semi-transparent fill onto a copy of the image

    Args:
        image   (Image.Image): RGB source image
        polygon (list):        COCO segmentation [[x1,y1,x2,y2,...], ...]
        color   (tuple):       RGB triple for both outline and fill
        width   (int):         outline stroke thickness in pixels

    Returns:
        Image.Image: RGBA image with polygon overlay composited on top
    """
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    for part in polygon:
        pts = [(int(part[i]), int(part[i + 1])) for i in range(0, len(part) - 1, 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=color + (ALPHA_FILL,))
            draw.line(pts + [pts[0]], fill=color + (255,), width=width)
    base = image.convert("RGBA")
    return Image.alpha_composite(base, overlay)


def _label_panel(image, text, color=(20, 20, 20)):
    """
    Returns a copy of image with a text caption strip drawn along the top

    Args:
        image (Image.Image): source RGBA or RGB image
        text  (str):         caption to render
        color (tuple):       RGB triple for caption text

    Returns:
        Image.Image: RGB copy with a white caption strip prepended
    """
    strip_h = 36
    canvas  = Image.new("RGB", (image.width, image.height + strip_h), (255, 255, 255))
    canvas.paste(image.convert("RGB"), (0, strip_h))
    draw    = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:  # Font not on path - fall back to PIL default bitmap
        font = ImageFont.load_default()
    draw.text((10, 8), text, fill=color, font=font)
    return canvas


def _draw_badge(draw, x, y, text):
    """Draws a small white-background caption with the polygon stats over a region centroid"""
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:  # Font not installed - fall back to PIL default bitmap
        font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    pad  = 2
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                   fill=(255, 255, 255, 220), outline=(20, 20, 20, 255))
    draw.text((x, y), text, fill=(20, 20, 20, 255), font=font)


def _polygon_centroid(polygon):
    """Returns (cx, cy) for the first polygon part, used to anchor stats badges"""
    if not polygon:
        return None
    first = polygon[0]
    xs    = first[0::2]
    ys    = first[1::2]
    if not xs or not ys:
        return None
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def _polygon_total_area(polygon):
    """Returns the combined pixel area of every part using the shoelace formula"""
    total = 0
    for part in polygon:
        xs = part[0::2]
        ys = part[1::2]
        if len(xs) < 3: continue  # Degenerate part
        s = 0
        for i in range(len(xs)):
            j  = (i + 1) % len(xs)
            s += xs[i] * ys[j] - xs[j] * ys[i]
        total += abs(s) // 2
    return int(total)


def _content_mask_panel(page, layers_paths):
    """
    Builds a panel showing each class's layer alpha tinted on top of the page

    Reads the three layer PNGs from disk and tints any opaque pixel with the
    class colour so the reviewer can verify the actual ink shape independent
    of polygon extraction quality

    Args:
        page          (Image.Image): RGB page background
        layers_paths  (dict):        keys hw/pr/ns mapping to absolute layer PNG paths

    Returns:
        Image.Image: RGBA page with class-tinted content overlay
    """
    base    = page.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for key, path in layers_paths.items():
        if not os.path.exists(path):
            continue
        alpha_arr = np.array(Image.open(path))[..., 3]
        mask_bool = alpha_arr > 0
        if not mask_bool.any():
            continue
        tint      = COLOR_CONTENT_BY_KEY.get(key, COLOR_FALLBACK)
        tint_arr  = np.zeros((alpha_arr.shape[0], alpha_arr.shape[1], 4), dtype=np.uint8)
        tint_arr[mask_bool, 0] = tint[0]
        tint_arr[mask_bool, 1] = tint[1]
        tint_arr[mask_bool, 2] = tint[2]
        tint_arr[mask_bool, 3] = ALPHA_CONTENT
        tint_img = Image.fromarray(tint_arr, "RGBA")
        overlay  = Image.alpha_composite(overlay, tint_img)
    return Image.alpha_composite(base, overlay)


def visualize_pages(output_root):
    """
    Emits one three-panel comparison per generated page

    Three panels stacked horizontally at source resolution:
      1. TEMPLATE polygons coloured by class
      2. REFINED polygons coloured by class with per-region stats badges
      3. CONTENT mask tinted per class - the source of truth for ink location

    Args:
        output_root (str): absolute path to data/synthetic/verify/polygons/{mode}

    Returns:
        int: number of comparison images written
    """
    dir_anno  = os.path.join(output_root, "annotations")
    dir_img   = os.path.join(output_root, "images")
    dir_layer = os.path.join(output_root, "layers")
    dir_vis   = os.path.join(output_root, "visualizations", "pages")
    os.makedirs(dir_vis, exist_ok=True)

    anno_files = sorted(glob.glob(os.path.join(dir_anno, "*.json")))
    if not anno_files:
        print(f"[!] No annotations found under {dir_anno}", flush=True)
        return 0

    written = 0
    for anno_path in anno_files:
        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_path = os.path.join(dir_img, data["image"])
        if not os.path.exists(img_path):
            print(f"[!] Missing page image for annotation {anno_path}", flush=True)
            continue

        page = Image.open(img_path).convert("RGB")
        stem = os.path.splitext(data["image"])[0]

        panel_tmpl = page.copy()
        panel_ref  = page.copy()

        for ann in data.get("annotations", []):
            cat       = ann.get("category", "")
            tmpl_poly = ann.get("template_polygon") or ann.get("gt_polygon")
            ref_poly  = ann.get("gt_polygon")
            color_t   = COLOR_TEMPLATE_BY_CAT.get(cat, COLOR_FALLBACK)
            color_r   = COLOR_REFINED_BY_CAT.get(cat, COLOR_FALLBACK)

            if tmpl_poly:
                panel_tmpl = _draw_polygon_overlay(panel_tmpl, tmpl_poly,
                                                    color_t, WIDTH_OUTLINE)
            if ref_poly:
                panel_ref = _draw_polygon_overlay(panel_ref, ref_poly,
                                                   color_r, WIDTH_OUTLINE)

        # Per-region badges on the refined panel only - too noisy to render on both
        ref_draw = ImageDraw.Draw(panel_ref)
        for ann in data.get("annotations", []):
            ref_poly = ann.get("gt_polygon")
            if not ref_poly: continue  # Nothing to anchor a badge against
            cx_cy = _polygon_centroid(ref_poly)
            if cx_cy is None: continue  # Degenerate polygon, skip rather than crash
            label = ann.get("subtype") or ann.get("category", "?")
            text  = f"{label} · {len(ref_poly)}p · {_polygon_total_area(ref_poly)}px"
            _draw_badge(ref_draw, cx_cy[0], cx_cy[1], text)

        layers_paths  = {k: os.path.join(dir_layer, f"{stem}_{k}.png") for k in ("hw", "pr", "ns")}
        panel_content = _content_mask_panel(page, layers_paths)

        panel_tmpl    = _label_panel(panel_tmpl,    "TEMPLATE polygons - Roboflow region (noise=red printed=orange handwritten=magenta)")
        panel_ref     = _label_panel(panel_ref,     "REFINED polygons - content-tight (noise=green printed=lime handwritten=cyan) with subtype·parts·area badges")
        panel_content = _label_panel(panel_content, "CONTENT mask - layer alpha tinted by class (ns=magenta pr=amber hw=blue) - source of truth")

        panels   = [panel_tmpl, panel_ref, panel_content]
        total_w  = sum(p.width for p in panels)
        canvas_h = panels[0].height
        combined = Image.new("RGB", (total_w, canvas_h), (255, 255, 255))
        x = 0
        for p in panels:
            combined.paste(p.convert("RGB"), (x, 0))
            x += p.width

        out_path = os.path.join(dir_vis, f"{stem}.compare.png")
        combined.save(out_path)
        written += 1

    print(f"[*] Page comparisons: {written} written to {dir_vis}", flush=True)
    return written


def _alpha_to_visual(alpha_arr):
    """
    Renders a single-channel alpha array as an RGB grayscale image for inspection

    Args:
        alpha_arr (np.ndarray): uint8 (H, W) alpha channel

    Returns:
        Image.Image: 3-channel grayscale image with the same dimensions
    """
    grayscale = np.stack([alpha_arr, alpha_arr, alpha_arr], axis=-1)
    return Image.fromarray(grayscale, "RGB")


def _composite_on_check(rgba):
    """
    Composites an RGBA asset on a light checkerboard so transparency is visible

    Args:
        rgba (Image.Image): RGBA asset to render against the checkerboard

    Returns:
        Image.Image: RGB image of asset composited on a 16-pixel check
    """
    w, h        = rgba.size
    bg          = Image.new("RGB", (w, h), (235, 235, 235))
    draw_bg     = ImageDraw.Draw(bg)
    tile        = 16
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                draw_bg.rectangle([x, y, x + tile, y + tile], fill=(210, 210, 210))
    bg.paste(rgba, (0, 0), rgba)
    return bg


def _find_manual_assets(root_dir):
    """
    Recursively enumerates every cleaned asset PNG under the manual asset tree

    Assets are alpha-cleaned at harvest time, so every PNG in the tree is
    already the final cleaned version  Skips any legacy .alpha_cache dirs

    Args:
        root_dir (str): top of the manual asset tree to scan

    Returns:
        list[str]: absolute paths to every .png asset file found
    """
    paths = []
    if not os.path.isdir(root_dir):
        return paths
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if ".alpha_cache" in dirpath:
            continue
        for fname in sorted(filenames):
            if fname.lower().endswith(".png"):
                paths.append(os.path.join(dirpath, fname))
    return paths


def visualize_assets(output_root, max_assets=80):
    """
    Emits per-asset diagnostic panels showing the cleaned alpha and its polygon envelope

    Assets are alpha-cleaned at harvest time, so this step walks the manual
    asset tree directly (no side-car cache required)

    Args:
        output_root (str): destination directory for the diagnostic panels
        max_assets  (int): cap on the number of panels to write so reviewers
            are not buried in thousands of files on large runs

    Returns:
        int: number of asset panels written
    """
    dir_vis = os.path.join(output_root, "visualizations", "assets")
    os.makedirs(dir_vis, exist_ok=True)

    asset_paths = _find_manual_assets(PATH_DIR_MANUAL_ROOT)
    if not asset_paths:
        print("[!] No manual assets found - run the harvesters first.", flush=True)
        return 0

    if len(asset_paths) > max_assets:
        random.shuffle(asset_paths)
        asset_paths = asset_paths[:max_assets]
        print(f"[*] Sampling {max_assets} assets from library for visualization", flush=True)

    written = 0
    for asset_path in asset_paths:
        try:
            asset = Image.open(asset_path).convert("RGBA")
        except Exception:
            print(f"[!] Unreadable asset: {asset_path}", flush=True)
            continue

        asset_arr    = np.array(asset)
        opaque       = int((asset_arr[..., 3] > 0).sum())

        kernel        = np.ones((SIZE_KERNEL_CLOSE, SIZE_KERNEL_CLOSE), np.uint8)
        closed_alpha  = cv2.morphologyEx(asset_arr[..., 3], cv2.MORPH_CLOSE, kernel)
        opaque_closed = int((closed_alpha > 0).sum())

        panel_img    = _label_panel(_composite_on_check(asset),
                                    "CLEANED - alpha set at harvest")
        panel_alpha  = _label_panel(_alpha_to_visual(asset_arr[..., 3]),
                                    f"Alpha channel ({opaque} px)")
        panel_closed = _label_panel(_alpha_to_visual(closed_alpha),
                                    f"Closed alpha ({opaque_closed} px) - polygon extractor view")

        rows    = [panel_img, panel_alpha, panel_closed]
        total_w = sum(p.width for p in rows)
        h       = rows[0].height
        canvas  = Image.new("RGB", (total_w, h), (255, 255, 255))
        x = 0
        for p in rows:
            canvas.paste(p, (x, 0))
            x += p.width

        name     = os.path.splitext(os.path.basename(asset_path))[0]
        out_path = os.path.join(dir_vis, f"{name}.compare.png")
        canvas.save(out_path)
        written += 1

    print(f"[*] Asset panels: {written} written to {dir_vis}", flush=True)
    return written


def main(count_arg, epoch_list, modes, workers):
    """
    Runs a small synthetic batch then renders the polygon and asset comparison panels

    Args:
        count_arg  (int):       number of layouts to generate per mode
        epoch_list (list[str]): epochs to draw templates from
        modes      (list[str]): generation modes to fan out across
        workers    (int):       ProcessPoolExecutor worker count
    """
    target_eras  = EPOCH_ALL if "all" in epoch_list else epoch_list
    target_modes = MODES_ALL if "all" in modes else modes

    print(f"[*] Verify polygons | Epochs: {target_eras} | Modes: {target_modes} | "
          f"Count: {count_arg} | Workers: {workers}", flush=True)

    tasks = _build_tasks(count_arg, target_eras, target_modes)
    if not tasks:
        return

    print(f"[*] Dispatching {len(tasks)} generation tasks", flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate_sample, task) for task in tasks]
        for future in as_completed(futures):
            try:
                info = future.result()
                completed += 1
                print(f"[{completed}/{len(tasks)}] Generated {info['name']} "
                      f"({info['mode']}) | Regions: {info['regions']} | "
                      f"Template: {info['template']}", flush=True)
            except Exception as e:  # Surface worker exceptions but keep draining
                print(f"[!] Worker error: {e}", flush=True)

    for mode in target_modes:
        mode_root = os.path.join(PATH_DIR_VERIFY_POLY, mode)
        visualize_pages(mode_root)
        visualize_assets(mode_root)

    print(f"[*] Verification complete | Browse {PATH_DIR_VERIFY_POLY}/{{mode}}/visualizations/",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify content-tight polygon refinement")
    parser.add_argument("--count",   type=int,   default=4)
    parser.add_argument("--epoch",   type=str,   nargs="+", default=["all"])
    parser.add_argument("--modes",   type=str,   nargs="+", default=["hybrid"],
                        choices=MODES_ALL + ["all"])
    parser.add_argument("--workers", type=int,   default=1)

    args = parser.parse_args()
    main(args.count, args.epoch, args.modes, args.workers)
