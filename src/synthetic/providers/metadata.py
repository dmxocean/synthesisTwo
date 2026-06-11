# -*- coding: utf-8 -*-
"""
Probabilistic ranking and retrieval engine for synthetic assets

This module implements the metadata provider responsible for discovering and querying synthetic assets from local catalogs. It uses a scoring-based retrieval mechanism to select the most appropriate snippets for a given polygonal region, considering dimensions, subtypes, and epoch preferences
"""

import os
import json
import random
import numpy as np

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class MetadataProvider:
    """
    Asset discovery and ranking provider

    This class maintains indices of available handwritten, printed, and noise assets. It provides a query interface that ranks potential matches based on geometric compatibility and metadata tags to ensure realistic document synthesis
    """

    def __init__(self, metadata_dir: str = "data/metadata"):
        """
        Initialize the provider and load all available metadata indices
        """
        self.metadata_dir = metadata_dir
        self.indices = self._load_all_indices()

    def _load_all_indices(self) -> dict:
        """
        Scan the metadata directory and deserialize all JSON catalogs
        """
        catalogs = {}
        if not os.path.exists(self.metadata_dir):
            return catalogs
        
        for f in os.listdir(self.metadata_dir):
            if f.endswith(".json"):
                name = f.replace(".json", "")
                with open(os.path.join(self.metadata_dir, f), "r", encoding="utf-8") as f_in:
                    catalogs[name] = json.load(f_in)
        return catalogs

    def query_assets(self, catalog_name: str, target_w: int, target_h: int, subtype: str = None, epoch: str = None, top_n: int = 15) -> dict:
        """
        Rank and retrieve a suitable asset from the specified catalog

        Args:
            catalog_name (str): Name of the catalog to query (e.g., manual_htr)
            target_w (int): Available width in the target region
            target_h (int): Available height in the target region
            subtype (str): Optional category filter for noise assets
            epoch (str): Optional epoch tag for soft-preference matching
            top_n (int): Number of top-ranked candidates to sample from
        Returns:
            dict: Metadata of the selected asset or None if no fit is found
        """
        pool = self.indices.get(catalog_name, [])
        if not pool:
            return None

        # Hard subtype filtering
        if catalog_name == "manual_noise" and subtype is not None:
            pool = [a for a in pool if a.get("subtype") == subtype]
            if not pool:
                return None

        is_noise = catalog_name == "manual_noise"
        target_aspect = target_w / target_h if target_h > 0 else 1.0
        candidates = []

        for asset in pool:
            aw = asset.get("dims", {}).get("w") or asset.get("geometry", {}).get("dims", {}).get("w")
            ah = asset.get("dims", {}).get("h") or asset.get("geometry", {}).get("dims", {}).get("h")

            if aw is None or ah is None:
                continue
                
            if aw > target_w * 1.15 or ah > target_h * 2.0:
                continue  # Admit row-end fits via loose aspect constraints

            score = 100.0

            # Width coverage scoring
            if target_w > 0:
                score += 40 * min(1.0, aw / target_w)

            # Orientation matching for noise artifacts
            if is_noise:
                a_aspect = aw / ah if ah > 0 else 1.0
                aspect_diff = abs(target_aspect - a_aspect)
                score -= min(30, aspect_diff * 15)

                solidity = asset.get("geometry", {}).get("solidity", 1.0)
                if 0.8 < target_aspect < 1.2 and solidity > 0.8:
                    score += 10  # Round stamp bonus

            # Metadata matching bonuses
            asset_epoch = asset.get("epoch")
            if epoch and asset_epoch and asset_epoch == epoch:
                score += 40
            if subtype and asset.get("subtype") == subtype:
                score += 30

            candidates.append((score, asset))

        if not candidates:
            fallback_pool = [a for a in pool if (a.get("dims", {}).get("w") or 0) < target_w]
            return random.choice(fallback_pool) if fallback_pool else random.choice(pool)

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_pool = candidates[:top_n]
        return random.choice(top_pool)[1]
