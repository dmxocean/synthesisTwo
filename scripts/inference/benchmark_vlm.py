# -*- coding: utf-8 -*-
"""
Performance benchmarking suite for Visual Language Models

This script evaluates different VLM profiles and RLSA kernel configurations against a set of historical PDF pages. It measures transcription accuracy, extraction quality, and processing throughput to identify optimal parameters for the production pipeline
"""

import os
import sys
import time
import json
import csv
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_PATH)

from src.preprocessing.pdf import DocumentProcessor
from src.preprocessing.normalize import preprocess_ctr_bgd_gs
from src.segmentation.inference import load_segmenter, predict_layers, COLORS
from src.detection.region.extract import extract_regions
from src.detection.vlm.qwen import QwenVLM
from src.detection.noise.predict import load_model as load_noise_model, predict_noise_instances
from src.core.gpu import DeviceManager
from src.core.confidence import LABEL_UNKNOWN

TEST_PDFS = [
    ("data/raw/1938/guiradbcn_a1938m1.pdf", 0),
]

COLOR_PR = (0, 255, 0)
COLOR_HW = (0, 0, 255)
COLOR_NS = (0, 150, 255)


def _precrop_regions(regs, img_np):
    """
    Generate PIL image crops for a collection of regions
    """
    return [(reg, Image.fromarray(img_np[reg.bbox[1]:reg.bbox[1] + reg.bbox[3],
                                         reg.bbox[0]:reg.bbox[0] + reg.bbox[2]]))
            for reg in regs]


def _vlm_thread(pr_regs, hw_regs, img_np, vlm):
    """
    Execute VLM transcription for printed and handwritten regions

    Args:
        pr_regs: Collection of printed regions
        hw_regs: Collection of handwritten regions
        img_np: Source page image as numpy array
        vlm: Initialized VLM instance
    Returns:
        Tuple containing printed results, handwritten results, and elapsed time
    """
    t0 = time.time()
    pr_crops = _precrop_regions(pr_regs, img_np)
    hw_crops = _precrop_regions(hw_regs, img_np)

    pr_results = []
    for reg, crop in pr_crops:
        text, conf, _ = vlm.transcribe_region(crop)
        pr_results.append({"id": reg.region_id, "bbox": reg.bbox,
                            "text": text.strip(), "conf": round(conf, 4)})

    hw_results = []
    for reg, crop in hw_crops:
        text, conf, _ = vlm.transcribe_region(crop)
        hw_results.append({"id": reg.region_id, "bbox": reg.bbox,
                            "text": text.strip(), "conf": round(conf, 4)})

    return pr_results, hw_results, time.time() - t0


def _resnet_thread(img_np, noise_model, device, prob_ns, k_ns):
    """
    Execute ResNet classification for noise instances

    Args:
        img_np: Source page image as numpy array
        noise_model: Initialized noise classification model
        device: Target compute device
        prob_ns: Noise probability map
        k_ns: RLSA kernel size for noise extraction
    Returns:
        Tuple containing detected noise instances and elapsed time
    """
    t0 = time.time()
    instances = predict_noise_instances(noise_model, device, img_np, prob_ns, kernel=k_ns)
    return instances, time.time() - t0


def _vlm_describe_unknown_noise(vlm, img_np, noise_instances):
    """
    Generate prose descriptions for uncertain noise instances via VLM

    This phase focuses exclusively on instances where the primary classifier returned an 'unknown' label
    Returns:
        Tuple containing updated instances and elapsed time
    """
    t0 = time.time()
    page_pil = Image.fromarray(img_np)
    for inst in noise_instances:
        if inst.mark_type == LABEL_UNKNOWN:
            x, y, w, h = inst.bbox
            desc, _ = vlm.describe_noise(page_pil.crop((x, y, x + w, y + h)))
            inst.description = desc
    return noise_instances, time.time() - t0


def _save_overlay(img_np, pr_results, hw_results, ns_results, out_path):
    """
    Render detection bounding boxes onto a visual overlay
    """
    canvas = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)
    for r in ns_results:
        x, y, w, h = r["bbox"]
        cv2.rectangle(canvas, (x, y), (x+w, y+h), COLOR_NS, 2)
    for r in hw_results:
        x, y, w, h = r["bbox"]
        cv2.rectangle(canvas, (x, y), (x+w, y+h), COLOR_HW, 2)
        cv2.putText(canvas, r["id"], (x, max(y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HW, 1)
    for r in pr_results:
        x, y, w, h = r["bbox"]
        cv2.rectangle(canvas, (x, y), (x+w, y+h), COLOR_PR, 2)
        cv2.putText(canvas, r["id"], (x, max(y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PR, 1)
    cv2.imwrite(out_path, canvas)


def _save_transcription(pr_results, hw_results, ns_results, out_path):
    """
    Format and save transcription results to a text file
    """
    lines = []

    lines.append("=" * 80)
    lines.append("PRINTED TEXT (PR)")
    lines.append("=" * 80)
    for r in pr_results:
        lines.append(f"\n> [{r['id']}]  bbox={r['bbox']}  conf={r['conf']:.4f}")
        lines.append(r["text"] if r["text"] else "[empty]")

    lines.append("\n")
    lines.append("=" * 80)
    lines.append("HANDWRITTEN TEXT (HW)")
    lines.append("=" * 80)
    for r in hw_results:
        lines.append(f"\n> [{r['id']}]  bbox={r['bbox']}  conf={r['conf']:.4f}")
        lines.append(r["text"] if r["text"] else "[empty]")

    lines.append("\n")
    lines.append("=" * 80)
    lines.append("NOISE MARKS (NS)")
    lines.append("=" * 80)
    for r in ns_results:
        is_unknown = r["type"] == LABEL_UNKNOWN
        desc_str = f'  → "{r["description"]}"' if (is_unknown and r.get("description")) else ""
        lines.append(
            f"[{r['id']}]  type={r['type']:<12}  conf={r['conf']:.4f}  bbox={r['bbox']}{desc_str}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _save_json_async(path, data):
    """
    Serialize results to JSON in a background thread
    """
    def _write():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    threading.Thread(target=_write, daemon=True).start()


def _profile_metrics(pr_results, hw_results, ns_results,
                     t_vlm, t_resnet, t_describe, wall_time):
    """
    Calculate performance and quality metrics for a specific benchmark profile
    """
    def _text_stats(results):
        if not results:
            return {"mean_conf": 0, "mean_chars": 0, "empty_ratio": 0}
        confs  = [r["conf"] for r in results]
        chars  = [len(r["text"]) for r in results]
        empty  = sum(1 for c in chars if c < 10)
        return {
            "mean_conf":   round(float(np.mean(confs)), 4),
            "mean_chars":  round(float(np.mean(chars)), 1),
            "empty_ratio": round(empty / len(results), 3),
        }

    pr_s = _text_stats(pr_results)
    hw_s = _text_stats(hw_results)
    ns_s = {"mean_conf": round(float(np.mean([r["conf"] for r in ns_results])), 4)
            if ns_results else 0}

    ns_unknown = sum(1 for r in ns_results if r["type"] == LABEL_UNKNOWN)
    total_wall  = wall_time + t_describe

    conf_score    = pr_s["mean_conf"]
    chars_score   = min(pr_s["mean_chars"] / 300.0, 1.0)
    empty_score   = 1.0 - pr_s["empty_ratio"]
    speed_score   = max(0.0, 1.0 - total_wall / 300.0)
    quality_score = round((conf_score * 0.4 + chars_score * 0.3 +
                           empty_score * 0.2 + speed_score * 0.1), 4)

    return {
        "pr_mean_conf":        pr_s["mean_conf"],
        "pr_mean_chars":       pr_s["mean_chars"],
        "pr_empty_ratio":      pr_s["empty_ratio"],
        "hw_mean_conf":        hw_s["mean_conf"],
        "hw_mean_chars":       hw_s["mean_chars"],
        "hw_empty_ratio":      hw_s["empty_ratio"],
        "ns_mean_conf":        ns_s["mean_conf"],
        "ns_unknown_count":    ns_unknown,
        "ns_classified_count": len(ns_results) - ns_unknown,
        "vlm_time":            round(t_vlm, 2),
        "resnet_time":         round(t_resnet, 2),
        "wall_phase1":         round(wall_time, 2),
        "t_describe":          round(t_describe, 2),
        "total_wall":          round(total_wall, 2),
        "quality_score":       quality_score,
    }


def benchmark_page(pdf_path, page_idx, seg_model, noise_model, vlm, device, profiles, output_root):
    """
    Process a single PDF page through all benchmark profiles
    """
    doc_id  = os.path.basename(pdf_path).replace(".pdf", "")
    out_dir = os.path.join(output_root, doc_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*100}")
    print(f"[*] BENCHMARK: {doc_id} (page {page_idx})")
    print(f"{'='*100}")

    img = DocumentProcessor.pdf_page_to_image(pdf_path, page_index=page_idx)
    if img is None:
        print(f"[!] Failed to load {pdf_path}")
        return []

    inp, _ = preprocess_ctr_bgd_gs(img)

    t_seg = time.time()
    probs = predict_layers(seg_model, inp, device)
    seg_time = time.time() - t_seg
    print(f"[*] Segmentation: {seg_time:.2f}s")

    overlay = img.astype(np.float32)
    for i in range(3):
        mask  = probs[i] > 0.2
        alpha = mask[..., None].astype(np.float32) * 0.5
        overlay = (1 - alpha) * overlay + alpha * COLORS[i]
    Image.fromarray(overlay.astype(np.uint8)).save(
        os.path.join(out_dir, "segmentation_composite.jpg"), quality=85)

    results  = {"pdf_path": pdf_path, "page_idx": page_idx,
                "preprocessing": "ctr_2.5_bgd_100_gs_1.0",
                "segmentation_time_seconds": seg_time, "profiles": []}
    csv_rows = []

    for name, k_pr, k_hw, k_ns in profiles:
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        print(f"\n{'-'*60}")
        print(f"[*] PROFILE: {name}  |  PR={k_pr}  HW={k_hw}  NS={k_ns}")

        pr_regs = extract_regions(probs[2], prefix="pr", kernel=k_pr, threshold=0.1)
        hw_regs = extract_regions(probs[1], prefix="hw", kernel=k_hw, threshold=0.1)
        print(f"    Regions: {len(pr_regs)} PR, {len(hw_regs)} HW")

        t_parallel = time.time()
        with ThreadPoolExecutor(max_workers=2) as exe:
            fut_vlm    = exe.submit(_vlm_thread, pr_regs, hw_regs, img, vlm)
            fut_resnet = exe.submit(_resnet_thread, img, noise_model, device, probs[0], k_ns)
            pr_results, hw_results, t_vlm = fut_vlm.result()
            ns_instances, t_resnet        = fut_resnet.result()
        wall_time = time.time() - t_parallel  # Phase 1 parallel wall time

        ns_instances, t_describe = _vlm_describe_unknown_noise(vlm, img, ns_instances)  # Phase 2 VLM description
        total_wall = wall_time + t_describe

        ns_results = [{"id": inst.mark_id, "bbox": inst.bbox,
                       "type": inst.mark_type, "conf": round(inst.confidence_score, 4),
                       "description": inst.description}
                      for inst in ns_instances]

        metrics = _profile_metrics(pr_results, hw_results, ns_results,
                                   t_vlm, t_resnet, t_describe, wall_time)

        print(f"    [REGIONS]  PR={len(pr_results)}  HW={len(hw_results)}  "
              f"NS={len(ns_results)}  (unknown={metrics['ns_unknown_count']}  "
              f"classified={metrics['ns_classified_count']})")
        print(f"    [QUALITY]  pr_conf={metrics['pr_mean_conf']:.3f}  "
              f"pr_chars={metrics['pr_mean_chars']:.0f}  "
              f"pr_empty={metrics['pr_empty_ratio']:.2f}  "
              f"score={metrics['quality_score']:.4f}")
        print(f"    [TIMING]   VLM_p1={t_vlm:.2f}s  ResNet={t_resnet:.2f}s  "
              f"Wall_p1={wall_time:.2f}s  Describe_p2={t_describe:.2f}s  "
              f"TOTAL={total_wall:.2f}s")

        overlay_path = os.path.join(out_dir, f"{safe_name}_detections.jpg")
        _save_overlay(img, pr_results, hw_results, ns_results, overlay_path)

        txt_path = os.path.join(out_dir, f"{safe_name}_transcription.txt")
        _save_transcription(pr_results, hw_results, ns_results, txt_path)

        profile_data = {
            "name": name,
            "kernels": {"PR": k_pr, "HW": k_hw, "NS": k_ns},
            "regions_count": {"PR": len(pr_results), "HW": len(hw_results), "NS": len(ns_results)},
            "metrics": metrics,
            "timing_seconds": {
                "vlm_thread":       t_vlm,
                "resnet_thread":    t_resnet,
                "wall_phase1":      wall_time,
                "describe_phase2":  t_describe,
                "total_wall":       total_wall,
                "sequential_equiv": t_vlm + t_resnet + t_describe,
            },
            "pr_regions": pr_results,
            "hw_regions": hw_results,
            "ns_regions": ns_results,
        }
        results["profiles"].append(profile_data)

        csv_rows.append({
            "pdf_id":              doc_id,
            "page_idx":            page_idx,
            "profile":             name,
            "kernel_pr":           str(k_pr),
            "kernel_hw":           str(k_hw),
            "kernel_ns":           str(k_ns),
            "pr_count":            len(pr_results),
            "hw_count":            len(hw_results),
            "ns_count":            len(ns_results),
            **metrics,
        })

    json_path = os.path.join(out_dir, f"benchmark_p{page_idx}.json")
    _save_json_async(json_path, results)
    print(f"\n[✓] JSON:           {json_path}")
    print(f"[✓] Transcriptions: {out_dir}/{{profile}}_transcription.txt")
    return csv_rows


if __name__ == "__main__":
    device          = DeviceManager.get_device()
    seg_model       = load_segmenter("unet", device)
    noise_model, _  = load_noise_model(device=device)
    vlm             = QwenVLM(device_map="auto")

    profiles = [
        ("Baseline OCR",       (25, 9),   (25, 9),   (25, 25)),
        ("Wide Lines",         (75, 9),   (40, 40),  (25, 25)),
        ("Extreme Wide Lines", (150, 9),  (40, 40),  (25, 25)),
        ("Small Paragraphs",   (75, 25),  (40, 40),  (25, 25)),
        ("Wide Small Paras",   (150, 25), (40, 40),  (25, 25)),
        ("Medium Paragraphs",  (75, 50),  (40, 40),  (40, 40)),
        ("Isotropic Medium",   (50, 50),  (40, 40),  (40, 40)),
        ("Large Blocks",       (100, 80), (50, 50),  (60, 60)),
        ("Isotropic Large",    (75, 75),  (50, 50),  (60, 60)),
    ]

    output_root = "outputs/debug/vlm"
    os.makedirs(output_root, exist_ok=True)

    all_csv_rows = []
    for pdf_path, page_idx in TEST_PDFS:
        if os.path.exists(pdf_path):
            rows = benchmark_page(pdf_path, page_idx, seg_model, noise_model, vlm,
                                  device, profiles, output_root)
            all_csv_rows.extend(rows)
        else:
            print(f"[!] PDF not found: {pdf_path}")

    csv_path = os.path.join(output_root, "profile_comparison.csv")  # Generate CSV results
    fieldnames = [
        "pdf_id", "page_idx", "profile", "kernel_pr", "kernel_hw", "kernel_ns",
        "pr_count", "hw_count", "ns_count",
        "pr_mean_conf", "pr_mean_chars", "pr_empty_ratio",
        "hw_mean_conf", "hw_mean_chars", "hw_empty_ratio",
        "ns_mean_conf", "ns_unknown_count", "ns_classified_count",
        "vlm_time", "resnet_time", "wall_phase1", "t_describe", "total_wall",
        "quality_score",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_csv_rows)

    print(f"\n{'='*110}")
    print(f"[*] PROFILE RANKING  (sorted by quality_score DESC)")
    print(f"{'='*110}")
    ranked = sorted(all_csv_rows, key=lambda r: r["quality_score"], reverse=True)
    print(f"{'Rank':<5} {'Profile':<22} {'PR':>5} {'HW':>5} {'NS':>5} "
          f"{'Unk':>5} {'PR conf':>8} {'PR chars':>9} {'Empty':>6} "
          f"{'P1(s)':>7} {'P2(s)':>7} {'Total(s)':>9} {'Score':>7}")
    print("-" * 105)
    for i, r in enumerate(ranked, 1):
        print(f"  {i:<4} {r['profile']:<22} {r['pr_count']:>5} {r['hw_count']:>5} "
              f"{r['ns_count']:>5} {r['ns_unknown_count']:>5} "
              f"{r['pr_mean_conf']:>8.3f} {r['pr_mean_chars']:>9.0f} "
              f"{r['pr_empty_ratio']:>6.2f} {r['wall_phase1']:>7.2f} "
              f"{r['t_describe']:>7.2f} {r['total_wall']:>9.2f} "
              f"{r['quality_score']:>7.4f}")

    if ranked:
        best = ranked[0]
        secs_page  = best["total_wall"]
        pages_pdf  = 50
        total_pdfs = 303
        secs_pdf   = secs_page * pages_pdf
        hours_all  = secs_page * pages_pdf * total_pdfs / 3600
        hours_30   = secs_page * pages_pdf * 30 / 3600

        print(f"\n{'='*110}")
        print(f"[*] THROUGHPUT ESTIMATE  -  best profile: {best['profile']}")
        print(f"{'='*110}")
        print(f"  Per page  : {secs_page:.1f}s  "
              f"(Phase1={best['wall_phase1']:.1f}s + Describe_p2={best['t_describe']:.1f}s)")
        print(f"  Per PDF   : ~{secs_pdf/60:.0f} min  ({pages_pdf} pages assumed)")
        print(f"  Full run  : ~{hours_all:.0f}h  ({total_pdfs} PDFs, "
              f"~{hours_all/8:.1f} days at 8h/day)")
        print(f"  Night run : ~{hours_30:.0f}h  (30 PDFs/batch)")
        print(f"  NS impact : {best['ns_unknown_count']} unknown/page → "
              f"{best['t_describe']:.2f}s describe overhead")

    print(f"\n[✓] profile_comparison.csv → {csv_path}")
    print(f"\n── Score formula ──────────────────────────────────────────────")
    print(f"  quality_score = 0.4×conf + 0.3×chars_score + 0.2×(1-empty) + 0.1×speed")
    print(f"  speed uses total_wall (Phase1 + Phase2)")

    time.sleep(1)
