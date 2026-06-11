# -*- coding: utf-8 -*-
"""
Train/val/test split over the synthetic factory pages.

Single source of truth: a materialised `split.json` keyed by source-page stem, so
all tiles of a page share one side (no tile-level leakage). Trainers consume
train+val; evaluators consume test. Assignment is a deterministic md5 hash of the
stem, so the file is identical across machines and runs.
"""

from __future__ import annotations

import os
import re
import json
import glob
import hashlib
from typing import Dict, List, Set, Union

from src.core.config import EPOCHS

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_SEED = 1337

_EXT_RE  = re.compile(r"\.(png|tiff?|jpe?g|json)$", re.IGNORECASE)
_TILE_RE = re.compile(r"_t\d+.*$")                       # _tNN + any trailing _input/_label
_TAIL_RE = re.compile(r"_(input|label|mask)$", re.IGNORECASE)


def epoch_of(name: str) -> str:
    """
    Identify the epoch encoded in a synthetic filename
    """
    parts = os.path.basename(name).split("_")
    if len(parts) >= 2 and parts[0] == "synth" and parts[1] in EPOCHS:
        return parts[1]
    return next((e for e in EPOCHS if e in name), "unknown")


def page_stem(name: str) -> str:
    """
    Collapse any tile filename to its source-page stem
    """
    base = _EXT_RE.sub("", os.path.basename(name))
    return _TAIL_RE.sub("", _TILE_RE.sub("", base))


def _assign(stem: str, ratios: Dict[str, float], seed: int) -> str:
    r = (int(hashlib.md5(f"{seed}:{stem}".encode()).hexdigest(), 16) % 10_000) / 10_000.0
    if r < ratios["test"]:
        return "test"
    if r < ratios["test"] + ratios["val"]:
        return "val"
    return "train"


def _as_list(images_dirs: Union[str, List[str]]) -> List[str]:
    return [images_dirs] if isinstance(images_dirs, str) else list(images_dirs)


def build_split(images_dirs, out_path, ratios=DEFAULT_RATIOS, seed=DEFAULT_SEED) -> Dict:
    """
    Assign every source page to train/val/test and write the manifest

    Args:
        images_dirs (Union[str, List[str]]): directories containing synthetic images
        out_path (str): destination path for the split JSON file
        ratios (Dict[str, float]): target ratios for train, val, and test splits
        seed (int): random seed for deterministic assignment
    Returns:
        Dict: the generated split manifest
    """
    stems = sorted({page_stem(p) for d in _as_list(images_dirs)
                    for p in glob.glob(os.path.join(d, "*_input.png"))})
    by_stem = {s: _assign(s, ratios, seed) for s in stems}
    manifest = {
        "seed": seed, "ratios": ratios, "n_pages": len(stems),
        "counts": {k: sum(v == k for v in by_stem.values()) for k in ("train", "val", "test")},
        "by_stem": by_stem,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def load_split(path: str) -> Dict:
    """
    Load a split manifest from disk
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_split(images_dirs, out_path, ratios=DEFAULT_RATIOS, seed=DEFAULT_SEED) -> Dict:
    """
    Load the manifest, building it once if absent

    Args:
        images_dirs (Union[str, List[str]]): directories containing synthetic images
        out_path (str): destination path for the split JSON file
        ratios (Dict[str, float]): target ratios for train, val, and test splits
        seed (int): random seed for deterministic assignment
    Returns:
        Dict: the loaded or generated split manifest
    """
    return load_split(out_path) if os.path.exists(out_path) else build_split(images_dirs, out_path, ratios, seed)


def stems_for(manifest: Dict, *splits: str) -> Set[str]:
    """
    Set of page stems belonging to the requested split(s)
    """
    want = set(splits)
    return {s for s, sp in manifest["by_stem"].items() if sp in want}
