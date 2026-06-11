# -*- coding: utf-8 -*-
"""
Research-grade figure helpers shared across stages

Saves PNG figures (training curves, uncertainty heatmaps, cluster scatters,
distribution charts) to disk so every stage can emit justifying visuals into
data/reports/

Headless Agg backend so it works on training nodes
No styling beyond figsize/grid to keep the look stable.

Pipeline role:
  training      : plot_series  (per-epoch metric curves)
  segmentation  : save_heatmap (per-pixel uncertainty heatmaps from prob maps)
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SIZE_FIG = (15, 8)


def plot_series(out_path, series_dict, xlabel="epoch", ylabel="value", title=""):
    """
    Plot one or more named series on shared axes and save as PNG

    Args:
        out_path (str): destination PNG path
        series_dict (dict): metric name to list of per-epoch values
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title (str): plot title
    """
    fig, ax = plt.subplots(figsize=SIZE_FIG)
    for label, values in series_dict.items():
        if not values: continue
        ax.plot(range(1, len(values) + 1), values, label=label, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_heatmap(array, out_path, title="", cmap="inferno", vmin=0.0, vmax=1.0):
    """
    Render a 2D array as a colour heatmap PNG

    Used for segmentation uncertainty maps where intense regions represent areas 
    of low model reliability. Written next to the page render so a researcher 
    can evaluate layer separation quality

    Args:
        array (np.ndarray): (H, W) float values to colour-map
        out_path (str): destination PNG path
        title (str): optional figure title
        cmap (str): matplotlib colormap name
        vmin (float): minimum colour range value
        vmax (float): maximum colour range value
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(array.shape[1] / 200.0 + 2, array.shape[0] / 200.0 + 1))
    im = ax.imshow(np.asarray(array), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
