# -*- coding: utf-8 -*-
"""
Per-page processing and record composition pipeline

This module implements the core workflow for turning a single page into validated archival records. It orchestrates segmentation, region extraction, VLM-based transcription for printed and handwritten text, and ResNet-based noise classification. The pipeline uses concurrent execution where possible to optimize GPU utilization
"""

import os
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Any

import numpy as np
from PIL import Image

from src.core.config import SEG_THRESHOLDS
from src.preprocessing.normalize import normalise_background, preprocess_ctr_bgd_gs
from src.preprocessing.pdf import DocumentProcessor
from src.segmentation.inference import predict_layers, save_layer_artifacts
from src.detection.region.extract import extract_regions
from src.detection.noise.predict import predict_noise_instances
from src.detection.records.builder import build_records

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_PATH, "data")
STORAGE_DIR = os.path.join(BASE_PATH, "storage")

IDX_NS, IDX_HW, IDX_PR = 0, 1, 2
ENGINE_VLM = "qwen3-vl"

@dataclasses.dataclass
class TranscribedRegion:
    """
    Data structure for a region after VLM transcription

    This class holds the spatial, textual, and confidence information for a detected text region, formatted for ingestion by the records builder
    """
    region_id: str
    bbox: List[int]
    full_text: str
    confidence_score: float
    reading_order: int = 0
    word_confidences: list = dataclasses.field(default_factory=list)
    engine: str = ENGINE_VLM

def transcribe_layer(vlm, page_np: np.ndarray, prob_channel: np.ndarray, prefix: str, threshold: float = None) -> List[TranscribedRegion]:
    """
    Extract and transcribe all regions within a specific semantic layer

    Args:
        vlm (Any): Initialized VLM instance for transcription
        page_np (np.ndarray): Original RGB image data for the page
        prob_channel (np.ndarray): Probability map for the target layer
        prefix (str): Identifier prefix for generated region IDs
        threshold (float): Binarization threshold for region extraction
    Returns:
        List[TranscribedRegion]: Collection of transcribed regions with confidence scores
    """
    kw = {} if threshold is None else {"threshold": threshold}
    out = []
    for rb in extract_regions(prob_channel, prefix=prefix, **kw):
        x, y, w, h = rb.bbox
        crop = Image.fromarray(page_np[y:y + h, x:x + w])
        text, conf, words = vlm.transcribe_region(crop)
        out.append(TranscribedRegion(
            region_id=rb.region_id,
            bbox=rb.bbox,
            full_text=text,
            confidence_score=conf,
            reading_order=rb.order,
            word_confidences=words
        ))
    return out

def _vlm_worker(vlm, page_np: np.ndarray, probs: np.ndarray) -> Tuple[List[TranscribedRegion], List[TranscribedRegion]]:
    """
    Concurrent worker for VLM transcription of multiple layers

    Args:
        vlm (Any): Initialized VLM instance
        page_np (np.ndarray): Original RGB image data
        probs (np.ndarray): Multi-channel probability map
    Returns:
        Tuple[List[TranscribedRegion], List[TranscribedRegion]]: Transcribed printed and handwritten regions
    """
    pr = transcribe_layer(vlm, page_np, probs[IDX_PR], "pr", threshold=SEG_THRESHOLDS[IDX_PR])
    hw = transcribe_layer(vlm, page_np, probs[IDX_HW], "hw", threshold=SEG_THRESHOLDS[IDX_HW])
    return pr, hw

def _resnet_worker(noise_model, device, page_np: np.ndarray, prob_ns: np.ndarray):
    """
    Concurrent worker for noise instance classification

    Args:
        noise_model (Any): Initialized ResNet model
        device (str): Execution device for the model
        page_np (np.ndarray): Original RGB image data
        prob_ns (np.ndarray): Probability map for the noise layer
    Returns:
        List[Any]: Classified noise instances
    """
    return predict_noise_instances(noise_model, device, page_np, prob_ns, threshold=SEG_THRESHOLDS[IDX_NS])

def _vlm_describe_noise(vlm, page_np: np.ndarray, noise_instances: List[Any]) -> List[Any]:
    """
    Generate prose descriptions for uncertain noise instances using the VLM

    Args:
        vlm (Any): Initialized VLM instance
        page_np (np.ndarray): Original RGB image data
        noise_instances (List[Any]): Collection of classified noise marks
    Returns:
        List[Any]: Noise instances updated with VLM descriptions where necessary
    """
    from src.core.confidence import LABEL_UNKNOWN
    page_pil = Image.fromarray(page_np)
    for inst in noise_instances:
        if inst.mark_type == LABEL_UNKNOWN:
            x, y, w, h = inst.bbox
            desc, _ = vlm.describe_noise(page_pil.crop((x, y, x + w, y + h)))
            inst.description = desc
    return noise_instances

def prefetch_page(pdf_path: str, page_idx: int):
    """
    Rasterize and preprocess a PDF page for the pipeline
    """
    page_np = DocumentProcessor.pdf_page_to_image(pdf_path, page_idx)
    norm, ink_mask = preprocess_ctr_bgd_gs(page_np)
    return page_np, norm, ink_mask

def index_page(page_np: np.ndarray, doc_id: str, page_id: str, image_path: str,
               seg_model: Any, vlm: Any, noise_model: Any, device: str,
               derived_dir: str = None, enhanced: bool = False):
    """
    Execute the full indexing pipeline for a single page

    Args:
        page_np (np.ndarray): Original RGB image data
        doc_id (str): Unique identifier for the document
        page_id (str): Unique identifier for the page
        image_path (str): Path to the source image file
        seg_model (Any): Initialized segmentation model
        vlm (Any): Initialized VLM model
        noise_model (Any): Initialized noise classification model
        device (str): Execution device for models
        derived_dir (str): Directory for saving intermediate artifacts
        enhanced (bool): Flag to enable advanced background normalization
    Returns:
        Tuple[Any, np.ndarray]: Generated records and the multi-channel probability map
    """
    if enhanced:
        norm, ink_mask = preprocess_ctr_bgd_gs(page_np)
    else:
        norm, ink_mask = normalise_background(page_np)

    probs = predict_layers(seg_model, norm, device)  # Generate layer probabilities

    with ThreadPoolExecutor(max_workers=2) as exe:
        fut_vlm = exe.submit(_vlm_worker, vlm, page_np, probs)
        fut_noise = exe.submit(_resnet_worker, noise_model, device, page_np, probs[IDX_NS])
        pr_regions, hw_regions = fut_vlm.result()
        noise = fut_noise.result()

    noise = _vlm_describe_noise(vlm, page_np, noise)  # Describe unknown marks

    records = build_records(
        document_id=doc_id,
        page_id=page_id,
        image_path=image_path,
        seg_pred=probs,
        ocr_pred=pr_regions,
        htr_pred=hw_regions,
        noise_pred=noise,
    )

    if derived_dir is not None:
        save_layer_artifacts(probs, page_np, ink_mask, page_id, derived_dir, use_ink_gate=not enhanced)

    return records, probs
