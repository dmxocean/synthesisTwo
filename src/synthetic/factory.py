# -*- coding: utf-8 -*-
"""
Synthetic document factory components

This module provides the reusable building blocks for document synthesis, including template discovery and loading mechanisms. It also defines the primary worker process responsible for compositing synthetic pages and managing the export of generated artifacts
"""

import os
import json
import random
import numpy as np

from src.synthetic.providers.metadata import MetadataProvider
from src.synthetic.generators.handwriting import HandwritingGenerator
from src.synthetic.generators.printed import PrintedGenerator
from src.synthetic.generators.noise import NoiseGenerator
from src.synthetic.core.assembler import DocumentAssembler
from src.synthetic.export.format import DataExporter
from src.synthetic.layouts.augment import augment_layout
from src.core.config import EPOCHS as EPOCH_ALL

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _list_templates(templates_root: str, epoch_list: list) -> list:
    """
    Retrieve sorted (epoch, template_path) tuples for all JSON templates

    This helper iterates through the provided epoch directories to discover and collect paths to all valid template files
    """
    result = []
    for epoch in epoch_list:
        epoch_dir = os.path.join(templates_root, epoch)
        if not os.path.isdir(epoch_dir):
            continue
        for fname in sorted(os.listdir(epoch_dir)):
            if fname.endswith(".json"):
                result.append((epoch, os.path.join(epoch_dir, fname)))
    return result

def _load_template(path: str) -> dict:
    """
    Load and deserialize a template JSON file from disk
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_sample(task: tuple) -> dict:
    """
    Execute a single synthetic document generation worker process

    This function orchestrates the full synthesis pipeline for one page, including metadata retrieval, component generation, layout augmentation, and final artifact persistence
    
    Args:
        task (tuple): Multi-parameter package containing epoch, template path, indices, seeds, and output configuration
    Returns:
        dict: Metadata regarding the generated sample, including region counts and number of valid tiles exported
    """
    (epoch, tpl_path, variant_idx, global_idx, seed, output_root, epoch_pool, mode) = task

    # Deterministic initialization
    random.seed(seed)
    np.random.seed(seed)

    # Component initialization
    metadata = MetadataProvider()
    hw_gen = HandwritingGenerator(metadata)
    pr_gen = PrintedGenerator(metadata)
    ns_gen = NoiseGenerator(metadata)
    assembler = DocumentAssembler(hw_gen, pr_gen, ns_gen)
    exporter = DataExporter(output_root)

    # Layout processing
    base_layout = _load_template(tpl_path)
    layout = augment_layout(base_layout, epoch_pool)
    name = f"synth_{epoch}_{global_idx:04d}_{variant_idx:03d}"

    # Document assembly and export
    img, layers, masks, annos = assembler.assemble(layout, mode=mode)
    tiles = exporter.save(name, img, layers, masks, annos)

    return {
        "name": name,
        "regions": len(layout["regions"]),
        "template": os.path.basename(tpl_path),
        "mode": mode,
        "epoch": epoch,
        "tiles": tiles,
    }
