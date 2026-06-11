# -*- coding: utf-8 -*-
"""
DBSCAN-based subset selection for demonstration

This script performs a live analysis of the data/raw directory to select a diverse subset of PDFs. It rasterizes initial pages, extracts MobileNet-v2 embeddings, and uses DBSCAN clustering to identify distinct document layouts across historical epochs
"""

import os
import sys
import json
import random
import collections

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import fitz
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_PATH)

from src.core.config import PATH_DIR_RAW, PATH_DIR_OUTPUTS, DICT_EPOCHS
from src.core.gpu import DeviceManager

PAGE_LIMIT = 5
PDFS_PER_EPOCH = 10
DPI_THUMB = 72
SEED = 42


class FeatureExtractor:
    """
    Image feature extraction using pre-trained MobileNet-v2
    """

    def __init__(self, device):
        """
        Initialize the extractor with a target compute device
        """
        self.device = device
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = nn.Identity()
        self.model = model.to(device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_features(self, img_pil):
        """
        Compute normalized embedding for a PIL image
        """
        tensor = self.tf(img_pil).unsqueeze(0).to(self.device)
        return self.model(tensor).squeeze().cpu().numpy()


def get_pdf_signature(pdf_path, extractor):
    """
    Calculate the mean visual embedding across the initial pages of a PDF

    Args:
        pdf_path (str): Path to the source PDF file
        extractor (FeatureExtractor): Initialized feature extraction instance
    Returns:
        np.ndarray: Averaged feature vector representing the document layout signature
    """
    features = []
    try:
        doc = fitz.open(pdf_path)
        n = min(len(doc), PAGE_LIMIT)
        mat = fitz.Matrix(DPI_THUMB / 72, DPI_THUMB / 72)
        for i in range(n):
            pix = doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            features.append(extractor.get_features(img))
        doc.close()
    except Exception as e:
        print(f"[!] Failed to process {pdf_path}: {e}")
        return None
    
    if not features:
        return None
    return np.mean(features, axis=0)


def _plot_clusters(epoch_plots, out_path):
    """
    Render DBSCAN clustering results for each historical epoch
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, epoch in zip(axes.flatten(), DICT_EPOCHS):
        if epoch not in epoch_plots:
            ax.axis("off")
            continue
        coords, labels = epoch_plots[epoch]
        for lbl in sorted(set(labels)):
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       label="noise" if lbl == -1 else f"c{lbl}", s=12, alpha=0.6)
        n_clusters = len(set(labels) - {-1})
        ax.set_title(f"{epoch} ({len(labels)} PDFs, {n_clusters} clusters)")
        ax.legend()
        ax.grid(alpha=0.25)
    plt.suptitle("DBSCAN Layout Clusters (PDF Signatures)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    """
    Orchestrate the diverse subset selection process via layout clustering
    """
    device = DeviceManager.get_device()
    extractor = FeatureExtractor(device)
    
    print(f"[*] Scanning {PATH_DIR_RAW}")
    buckets = collections.defaultdict(list)
    for year_dir in sorted(os.listdir(PATH_DIR_RAW)):
        if not year_dir.isdigit():
            continue
        year = int(year_dir)
        epoch = next((name for name, r in DICT_EPOCHS.items() if year in r), None)
        if not epoch:
            continue
        
        dir_path = os.path.join(PATH_DIR_RAW, year_dir)
        pdfs = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]
        buckets[epoch].extend(pdfs)

    final_selection = {}
    epoch_plots = {}
    output_dir = os.path.join(PATH_DIR_OUTPUTS, "indexing")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "subset.json")

    for epoch, pdf_paths in buckets.items():
        print(f"\n[*] Processing {epoch} ({len(pdf_paths)} PDFs)")
        
        signatures = []
        valid_paths = []
        for p in tqdm(pdf_paths, desc=f"Extracting {epoch}"):
            sig = get_pdf_signature(p, extractor)
            if sig is not None:
                signatures.append(sig)
                valid_paths.append(p)

        if not signatures:
            continue

        X = np.array(signatures)  # Layout clustering via DBSCAN
        pca = PCA(n_components=min(50, len(X)), random_state=SEED)
        X_reduced = pca.fit_transform(X)
        
        pca_2d = PCA(n_components=2, random_state=SEED)
        X_2d = pca_2d.fit_transform(X)

        if len(signatures) < PDFS_PER_EPOCH:
            final_selection[epoch] = valid_paths
            epoch_plots[epoch] = (X_2d, np.zeros(len(X_2d)))
            continue

        from sklearn.neighbors import NearestNeighbors
        nn_model = NearestNeighbors(n_neighbors=min(5, len(X_reduced))).fit(X_reduced)
        distances, _ = nn_model.kneighbors(X_reduced)
        eps = np.percentile(distances[:, -1], 15)  # Use 15th percentile of k-NN distance for eps
        eps = max(eps, 0.1)

        dbscan = DBSCAN(eps=eps, min_samples=2).fit(X_reduced)
        labels = dbscan.labels_
        epoch_plots[epoch] = (X_2d, labels)
        
        selected = []  # Diversity-aware selection strategy
        unique_labels = sorted(set(labels))
        clusters = [l for l in unique_labels if l != -1]
        
        cluster_idx = 0
        while len(selected) < PDFS_PER_EPOCH and (clusters or -1 in unique_labels):
            if clusters:
                cid = clusters[cluster_idx % len(clusters)]
                indices = np.where(labels == cid)[0]
            else:
                indices = np.where(labels == -1)[0]
            
            for idx in indices:
                path = valid_paths[idx]
                if path not in selected:
                    selected.append(path)
                    break
            
            cluster_idx += 1
            if not clusters and -1 in unique_labels:
                break
            if cluster_idx > 1000:
                break

        if len(selected) < PDFS_PER_EPOCH:
            remaining = [p for p in valid_paths if p not in selected]
            random.seed(SEED)
            selected.extend(random.sample(remaining, min(len(remaining), PDFS_PER_EPOCH - len(selected))))

        final_selection[epoch] = selected
        print(f"[*] {epoch}: Selected {len(selected)} PDFs from {len(clusters)} clusters (noise={list(labels).count(-1)})")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_selection, f, indent=2, ensure_ascii=False)
    
    _plot_clusters(epoch_plots, os.path.join(output_dir, "clusters.png"))
    
    print(f"\n[✓] Success! Manifest saved to: {output_path}")
    print(f"[✓] Plot saved to: {os.path.join(output_dir, 'clusters.png')}")


if __name__ == "__main__":
    main()
