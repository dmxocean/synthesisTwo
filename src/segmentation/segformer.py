# -*- coding: utf-8 -*-
"""
SegFormer segmentation model for document processing

This module provides a SegFormer architecture with a MiT-B3 encoder and an MLP decoder. It is configured to output three independent channels for multi-label segmentation of noise, handwriting, and printed text
"""

import segmentation_models_pytorch as smp
from src.segmentation.data import COUNT_CLASSES  # Shared class count constant

NAME_ENCODER = "mit_b3"  # MiT encoder architecture
CHANNELS_INPUT = 3  # RGB input channels

def build_segformer(pretrained=True):
    """
    Construct a SegFormer model for the multi-label task

    The model utilizes an ImageNet-pretrained encoder by default and returns raw logits to maintain numerical stability during training
    Args:
        pretrained (bool): Whether to load ImageNet encoder weights
    Returns:
        torch.nn.Module: SegFormer model outputting (B, 3, H, W) raw logits
    """
    return smp.Segformer(
        encoder_name=NAME_ENCODER,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=CHANNELS_INPUT,
        classes=COUNT_CLASSES,
        activation=None,  # Raw logits for stable BCE+Dice calculation
    )
