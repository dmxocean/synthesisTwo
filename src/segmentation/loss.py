# -*- coding: utf-8 -*-
"""
Multi-label loss and metric helpers for the 3-channel document segmenter

This module combines BCEWithLogitsLoss with multi-label Dice loss to optimize per-pixel gradients and region overlap. The architecture supports independent channel prediction, allowing pixels to belong to multiple classes simultaneously
"""

import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss

EPS_IOU = 1e-6  # Epsilon to prevent division by zero in IoU
WEIGHT_BCE_DEFAULT = 0.5  # Default weight for Binary Cross Entropy
WEIGHT_DICE_DEFAULT = 0.5  # Default weight for Dice loss
COUNT_CLASSES = 3  # Number of segmentation classes

def iou_per_class(logits, target, num_classes=COUNT_CLASSES):
    """
    Compute per-class intersection over union (IoU) metrics

    The calculation applies a sigmoid activation followed by hard thresholding at 0.5 to determine class occupancy
    Args:
        logits (torch.Tensor): Raw model outputs (B, 3, H, W)
        target (torch.Tensor): Multi-label ground truth (B, 3, H, W)
        num_classes (int): Number of independent classes
    Returns:
        torch.Tensor: Scalar IoU values for each class (3,)
    """
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    target = target.float()

    ious = torch.zeros(num_classes, device=logits.device)
    for c in range(num_classes):
        pred_c = pred[:, c, :, :]
        target_c = target[:, c, :, :]
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - inter
        ious[c] = inter / (union + EPS_IOU)
    return ious

class MultiLabelLoss(nn.Module):
    """
    Hybrid loss combining BCE and multi-label Dice for independent channel segmentation

    This loss treats the segmentation task as simultaneous binary tasks, enabling the network to learn overlapping patterns without class competition
    """

    def __init__(self,
                 weight_bce=WEIGHT_BCE_DEFAULT,
                 weight_dice=WEIGHT_DICE_DEFAULT):
        """
        Initialize the multi-label loss with specific component weights

        Args:
            weight_bce (float): Weight for the BCE component
            weight_dice (float): Weight for the Dice component
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Stable BCE with internal sigmoid
        self.dice = DiceLoss(mode="multilabel", from_logits=True)  # Dice for multi-label regions
        self.w_bce = weight_bce
        self.w_dice = weight_dice

    def forward(self, logits, target):
        """
        Compute the weighted combination of BCE and Dice losses

        Args:
            logits (torch.Tensor): Predicted logits (B, 3, H, W)
            target (torch.Tensor): Target binary masks (B, 3, H, W)
        Returns:
            torch.Tensor: Scalar loss value
        """
        target = target.float()  # Convert target to float for loss calculation
        loss_bce = self.bce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.w_bce * loss_bce + self.w_dice * loss_dice
