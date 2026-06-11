# -*- coding: utf-8 -*-
"""
ResNet-18 patch classifier for noise subtype detection

Maps each RGB crop from the noise layer into one of five subtypes:
{stamps, circles, crosses, lines, marks}. Uses a timm ImageNet-pretrained
ResNet-18 with the final FC layer replaced by a 5-class head. Inference
input is a 224×224 RGB crop from the original page RGB image - the crop
is taken at the bounding box coordinates produced by bbox_extract.py

Pipeline role:
  Inputs  224×224 RGB crop from original page
  Outputs mark_type (str) and confidence_score (float) per crop

Critical design decisions:
  The classifier reads RGB crops, not mask crops. This means it sees the full
  visual texture of the mark (colour gradients, stroke patterns) which is
  essential when the binary mask is fragmentary for faded historical marks
"""

import os
import torch
import torch.nn as nn
import timm

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CLASSES_NOISE  = ["stamps", "circles", "crosses", "lines", "marks"]
COUNT_CLASSES  = len(CLASSES_NOISE)
SIZE_INPUT     = 224  # Timm ResNet-18 default input


def build_classifier(pretrained=True):
    """
    Return a 5-class ResNet-18 for noise subtype classification

    Args:
        pretrained (bool): load ImageNet weights for the backbone
    Returns:
        nn.Module: model whose forward returns (B, 5) raw logits
    """
    model = timm.create_model("resnet18", pretrained=pretrained, num_classes=COUNT_CLASSES)
    return model


def predict_crop(model, crop_tensor, device):
    """
    Run one crop through the classifier and return mark_type + confidence

    Args:
        model (nn.Module): trained classifier
        crop_tensor (torch.Tensor): (1, 3, 224, 224) normalised float tensor
        device (torch.device): inference device
    Returns:
        Tuple[str, float]: mark_type and confidence score
    """
    model.eval()
    with torch.no_grad():
        logits = model(crop_tensor.to(device))
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = int(probs.argmax())
    return CLASSES_NOISE[idx], float(probs[idx].item())
