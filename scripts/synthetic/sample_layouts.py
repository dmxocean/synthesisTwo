# -*- coding: utf-8 -*-
"""
Proportional page sampler for layout discovery across GUIRAD epochs

Uses DBSCAN clustering on MobileNet-v2 embeddings to identify natural layout
groups within each historical epoch, then selects pages proportionally from
each cluster for manual annotation in Roboflow
"""

import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import fitz
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from src.core.config import DICT_EPOCHS
from src.core.gpu import DeviceManager

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIR_RAW = os.path.join(PATH_ROOT, "data", "raw")
PATH_DIR_SAMPLES = os.path.join(PATH_ROOT, "data", "interim", "layouts", "samples")

# Constants
SEED_RANDOM          = 42
SIZE_QUOTA_EPOCH     = 250
SIZE_MAX_PAGES_PDF   = 15
SIZE_MAX_PDFS_EPOCH  = 300
SIZE_THUMB           = (224, 224)
VAL_DPI_THUMB        = 72
VAL_DPI_EXPORT       = 300
VAL_MIN_SAMPLES_DBSCAN = 5
VAL_SHARE_NOISE      = 0.10
NUM_WORKERS          = 4

SIZE_A4   = (2480, 3508)
RATIO_A4  = 2480 / 3508

_worker_extractor = None  # Worker-level global set via _init_worker


def _init_worker(device_str):
    """Loads MobileNet-v2 once per worker process to avoid re-loading per task"""
    global _worker_extractor
    _worker_extractor = FeatureExtractor(torch.device(device_str))


class FeatureExtractor:
    """MobileNet-v2 backbone returning a 1280-dim embedding per page thumbnail"""
    def __init__(self, device):
        self.device = device
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = nn.Identity()
        self.model = model.to(device).eval()
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, img_np):
        pil = Image.fromarray(img_np).resize(SIZE_THUMB, Image.LANCZOS)
        tensor = self.tf(pil).unsqueeze(0).to(self.device)
        return self.model(tensor).squeeze().cpu().numpy()


def _standardize_a4(img_np):
    """Center-crops to A4 aspect ratio then resizes to 2480x3508"""
    h, w = img_np.shape[:2]
    if (w / h) > RATIO_A4:
        target_w = int(h * RATIO_A4)
        start_x  = (w - target_w) // 2
        img_np   = img_np[:, start_x:start_x + target_w]
    else:
        target_h = int(w / RATIO_A4)
        start_y  = (h - target_h) // 2
        img_np   = img_np[start_y:start_y + target_h, :]
    return cv2.resize(img_np, SIZE_A4, interpolation=cv2.INTER_LANCZOS4)


def _process_pdf(args):
    """
    Worker: rasterizes up to SIZE_MAX_PAGES_PDF pages at VAL_DPI_THUMB and
    extracts MobileNet-v2 embeddings  Uses _worker_extractor set by _init_worker
    """
    pdf_path, epoch, path_root, max_pages = args
    global _worker_extractor
    features, meta = [], []
    try:
        doc = fitz.open(pdf_path)
        n_pages = min(len(doc), max_pages)
        mat = fitz.Matrix(VAL_DPI_THUMB / 72, VAL_DPI_THUMB / 72)
        for pg in range(n_pages):
            try:
                pix = doc[pg].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()
            except Exception:
                continue
            features.append(_worker_extractor(img))
            meta.append({
                "pdf":   os.path.relpath(pdf_path, path_root).replace("\\", "/"),
                "stem":  os.path.splitext(os.path.basename(pdf_path))[0],
                "page":  pg,
                "epoch": epoch,
            })
        doc.close()
    except Exception:
        pass  # Silently skip corrupt PDFs
    return features, meta


def _render_export_page(args):
    """
    Worker: renders one page at VAL_DPI_EXPORT, standardizes to A4, writes PNG
    Returns manifest entry dict or None on failure
    """
    abs_pdf, page_index, out_path, epoch, cluster_id, cluster_size, rel_pdf = args
    try:
        doc = fitz.open(abs_pdf)
        mat = fitz.Matrix(VAL_DPI_EXPORT / 72, VAL_DPI_EXPORT / 72)
        pix = doc[page_index].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()
        doc.close()
        img = _standardize_a4(img)
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return {
            "file":         f"{epoch}/{os.path.basename(out_path)}",
            "source":       rel_pdf,
            "page":         page_index,
            "epoch":        epoch,
            "cluster_id":   cluster_id,
            "cluster_size": cluster_size,
        }
    except Exception as e:
        print(f"[!] Render failed: {os.path.basename(out_path)}: {e}")
        return None


def _cache_file(samples_dir, epoch):
    return os.path.join(samples_dir, ".cache", f"features_{epoch}.npz")


def _load_epoch_cache(samples_dir, epoch, current_pdfs):
    path = _cache_file(samples_dir, epoch)
    if not os.path.exists(path):
        return None, None
    try:
        data = np.load(path, allow_pickle=True)
        if sorted(json.loads(str(data["source_pdfs"]))) != sorted(current_pdfs):
            print(f"[*] {epoch}: cache miss (PDF set changed), re-extracting")
            return None, None
        features = data["features"]
        meta = json.loads(str(data["meta_json"]))
        print(f"[*] {epoch}: loaded {len(features)} embeddings from cache")
        return features, meta
    except Exception as e:
        print(f"[*] {epoch}: cache unreadable ({e}), re-extracting")
        return None, None


def _save_epoch_cache(samples_dir, epoch, features, meta, source_pdfs):
    path = _cache_file(samples_dir, epoch)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        features=features.astype(np.float32),
        meta_json=np.array(json.dumps(meta)),
        source_pdfs=np.array(json.dumps(sorted(source_pdfs))),
    )


def _bucket_pdfs(path_raw):
    """Walks raw directory and assigns each PDF to its historical epoch"""
    if not os.path.isdir(path_raw):
        raise FileNotFoundError(f"Missing {path_raw}")
    buckets = {name: [] for name in DICT_EPOCHS}
    for year_dir in sorted(os.listdir(path_raw)):
        if not year_dir.isdigit():
            continue
        year  = int(year_dir)
        epoch = next((name for name, r in DICT_EPOCHS.items() if year in r), None)
        if epoch is None:
            continue
        base = os.path.join(path_raw, year_dir)
        buckets[epoch] += sorted(
            os.path.join(base, f)
            for f in os.listdir(base)
            if f.lower().endswith(".pdf")
        )
    rng = random.Random(SEED_RANDOM)
    for epoch in buckets:
        paths = sorted(buckets[epoch])
        if len(paths) > SIZE_MAX_PDFS_EPOCH:
            paths = rng.sample(paths, SIZE_MAX_PDFS_EPOCH)
        buckets[epoch] = paths
    return buckets


def _scan_epoch(epoch, paths, samples_dir, device_str):
    """
    Extracts features for all PDFs in the epoch using NUM_WORKERS parallel processes
    Results are cached - reruns skip extraction if the PDF set is unchanged
    """
    features_c, meta_c = _load_epoch_cache(samples_dir, epoch, paths)
    if features_c is not None:
        return features_c, meta_c

    args = [(p, epoch, PATH_ROOT, SIZE_MAX_PAGES_PDF) for p in paths]
    all_features, all_meta = [], []

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(device_str,),
    ) as pool:
        futures = {pool.submit(_process_pdf, a): a[0] for a in args}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"[*] {epoch} {done}/{len(futures)} PDFs processed", flush=True)
            feats, meta = fut.result()
            all_features.extend(feats)
            all_meta.extend(meta)

    if not all_features:
        return np.array([]), []

    features = np.array(all_features)
    _save_epoch_cache(samples_dir, epoch, features, all_meta, paths)
    return features, all_meta


def _auto_eps(reduced, k):
    """90th-percentile k-NN distance as a conservative DBSCAN epsilon"""
    nbrs  = NearestNeighbors(n_neighbors=k).fit(reduced)
    dists, _ = nbrs.kneighbors(reduced)
    return float(np.percentile(dists[:, -1], 90))


def _cluster(features, epoch):
    """PCA reduction followed by DBSCAN clustering"""
    n       = len(features)
    reduced = PCA(n_components=min(50, n - 1), random_state=SEED_RANDOM).fit_transform(features)
    eps     = _auto_eps(reduced, VAL_MIN_SAMPLES_DBSCAN)
    labels  = DBSCAN(eps=eps, min_samples=VAL_MIN_SAMPLES_DBSCAN, n_jobs=-1).fit_predict(reduced)
    n_clusters = len(set(labels) - {-1})
    n_noise    = int(np.sum(labels == -1))
    print(f"[*] {epoch:<15} eps={eps:.3f} clusters={n_clusters} noise={n_noise}/{n} ({100 * n_noise / n:.0f}%)")
    coords_2d = PCA(n_components=2, random_state=SEED_RANDOM).fit_transform(features)
    return labels, coords_2d


def _proportional_sample(features, meta, labels, quota):
    """Selects quota pages from DBSCAN clusters with proportional allocation"""
    if len(meta) <= quota:
        return [m | {"cluster_id": 0, "cluster_size": len(meta)} for m in meta]

    rng        = np.random.default_rng(SEED_RANDOM)
    noise_idx  = np.where(labels == -1)[0]
    cluster_ids = sorted(set(labels) - {-1})

    if not cluster_ids:
        chosen = rng.choice(len(meta), size=quota, replace=False)
        return [meta[i] | {"cluster_id": 0, "cluster_size": len(meta)} for i in chosen]

    selected = []

    noise_quota = min(len(noise_idx), int(round(quota * VAL_SHARE_NOISE)))
    if noise_quota > 0:
        for i in rng.choice(noise_idx, size=noise_quota, replace=False):
            selected.append(meta[i] | {"cluster_id": -1, "cluster_size": len(noise_idx)})

    remaining   = quota - len(selected)
    n_clustered = len(labels) - len(noise_idx)
    slots = {}
    for cid in cluster_ids:
        size  = int(np.sum(labels == cid))
        share = size / max(n_clustered, 1)
        slots[cid] = max(1, int(round(remaining * share)))

    diff    = remaining - sum(slots.values())
    ordered = sorted(slots, key=slots.get, reverse=True)
    for cid in ordered:
        if diff == 0:
            break
        adj = 1 if diff > 0 else -1
        if adj < 0 and slots[cid] <= 1:
            continue
        slots[cid] += adj
        diff -= adj

    for cid in cluster_ids:
        idx      = np.where(labels == cid)[0]
        slot     = min(slots[cid], len(idx))
        centroid = features[idx].mean(axis=0)
        order    = np.argsort(np.linalg.norm(features[idx] - centroid, axis=1))
        for i in idx[order[:slot]]:
            selected.append(meta[i] | {"cluster_id": int(cid), "cluster_size": len(idx)})

    rng.shuffle(selected)
    return selected[:quota]


def _export(selected, samples_dir, epoch):
    """Renders selected pages at VAL_DPI_EXPORT using NUM_WORKERS parallel workers"""
    epoch_dir = os.path.join(samples_dir, epoch)
    os.makedirs(epoch_dir, exist_ok=True)

    work = []
    for item in selected:
        fname   = f"{item['stem']}__p{item['page']:03d}.png"
        out     = os.path.join(epoch_dir, fname)
        abs_pdf = os.path.join(PATH_ROOT, item["pdf"].replace("/", os.sep))
        work.append((
            abs_pdf, item["page"], out,
            epoch, item["cluster_id"], item["cluster_size"],
            item["pdf"],
        ))

    entries = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for entry in tqdm(pool.map(_render_export_page, work),
                          total=len(work), desc=f"[*] Exporting {epoch}"):
            if entry is not None:
                entries.append(entry)
    return entries


def _plot(epoch_plots, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, epoch in zip(axes.flatten(), DICT_EPOCHS):
        if epoch not in epoch_plots:
            ax.axis("off")
            continue
        coords, labels = epoch_plots[epoch]
        for lbl in sorted(set(labels)):
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       label="noise" if lbl == -1 else f"c{lbl}", s=8, alpha=0.5)
        n_clusters = len(set(labels) - {-1})
        ax.set_title(f"{epoch} ({len(labels)} pages, {n_clusters} clusters)")
        ax.legend()
        ax.grid(alpha=0.25)
    plt.suptitle("DBSCAN layout clusters by epoch")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run(path_raw, samples_dir, quota):
    random.seed(SEED_RANDOM)
    np.random.seed(SEED_RANDOM)
    os.makedirs(samples_dir, exist_ok=True)

    manifest_path = os.path.join(samples_dir, "manifest.json")
    existing = []
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            existing = json.load(f)

    done = {}
    for entry in existing:
        done.setdefault(entry["epoch"], set()).add((entry["source"], entry["page"]))

    needed_per_epoch = {ep: quota - len(done.get(ep, set())) for ep in DICT_EPOCHS}
    if all(n <= 0 for n in needed_per_epoch.values()):
        print(f"[*] All epochs already at quota ({quota} pages)")
        return

    for ep, n in needed_per_epoch.items():
        print(f"[*] {ep:<15} {'+' + str(n) + ' needed' if n > 0 else 'at quota'}")

    device     = DeviceManager.get_device()
    DeviceManager.print_hardware_summary()
    device_str = str(device)

    print("[*] Bucketing PDFs by epoch")
    buckets = _bucket_pdfs(path_raw)
    for ep, paths in buckets.items():
        print(f"[*] {ep:<15} {len(paths):>4} PDFs")

    print(f"[*] Extracting page features ({NUM_WORKERS} workers)")
    epoch_data = {}
    for epoch, paths in buckets.items():
        if needed_per_epoch.get(epoch, 0) <= 0:
            continue
        if not paths:
            print(f"[!] {epoch}: no PDFs found")
            continue
        print(f"[*] Scanning {epoch}...")
        feats, meta = _scan_epoch(epoch, paths, samples_dir, device_str)
        if len(feats) == 0:
            print(f"[!] {epoch}: no readable pages")
            continue
        already = done.get(epoch, set())
        if already:
            keep = [(f, m) for f, m in zip(feats, meta)
                    if (m["pdf"], m["page"]) not in already]
            if not keep:
                print(f"[!] {epoch}: no new pages after exclusion")
                continue
            feats_k, meta_k = zip(*keep)
            feats, meta = np.array(feats_k), list(meta_k)
        epoch_data[epoch] = (feats, meta)
        print(f"[*] {epoch:<15} {len(feats):>4} new pages available")

    print("[*] Running DBSCAN per epoch")
    epoch_labels = {}
    epoch_plots  = {}
    for epoch, (feats, meta) in epoch_data.items():
        labels, coords_2d = _cluster(feats, epoch)
        epoch_labels[epoch] = labels
        epoch_plots[epoch]  = (coords_2d, labels)

    print(f"[*] Selecting and exporting pages ({NUM_WORKERS} workers)")
    new_entries = []
    for epoch, (feats, meta) in epoch_data.items():
        labels   = epoch_labels[epoch]
        need     = min(needed_per_epoch[epoch], len(meta))
        selected = _proportional_sample(feats, meta, labels, need)
        print(f"[*] {epoch:<15} {len(selected):>4} pages selected")
        entries  = _export(selected, samples_dir, epoch)
        new_entries.extend(entries)

    updated = existing + new_entries
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f"[*] Manifest: {len(existing)} existing + {len(new_entries)} new = {len(updated)} total")

    if epoch_plots:
        _plot(epoch_plots, os.path.join(samples_dir, "clusters.png"))
    print(f"[*] {len(new_entries)} new pages exported to {samples_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBSCAN layout sampler")
    parser.add_argument("--raw",    default=PATH_DIR_RAW, help="Path to data/raw/")
    parser.add_argument("--output", default=PATH_DIR_SAMPLES, help="Destination for exported PNGs and manifest")
    parser.add_argument("--quota",  type=int, default=250, help="Target pages per epoch; re-run with a higher value to append more")
    args = parser.parse_args()
    run(args.raw, args.output, args.quota)
