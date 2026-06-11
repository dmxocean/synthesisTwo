# -*- coding: utf-8 -*-
"""
Multi-label 3-class dataset for the document segmenter

This module pairs synthetic pages with multi-frame TIFF labels to produce binary masks for noise, handwriting, and printed text. It handles background normalization and data augmentation for training robust segmentation models
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.preprocessing.normalize import normalise_background

SIZE_CROP = 768  # Square crop fed to the network
COUNT_CLASSES = 3  # Noise, handwriting, printed
IDX_NS = 0  # Noise frame index
IDX_HW = 1  # Handwriting frame index
IDX_PR = 2  # Printed text frame index
IDX_INTER = 3  # Discarded interaction frame
KEY_LAYERS = ("ns", "hw", "pr")
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
PROB_AUG_ROTATE = 0.5
LIMIT_DEG_ROTATE = 2
LIMIT_BRIGHTNESS = 0.15
LIMIT_CONTRAST = 0.15
PROB_AUG_NOISE = 0.3
RANGE_STD_GAUSS = (0.02, 0.10)
PROB_AUG_BLUR = 0.3
LIMIT_BLUR_KERNEL = (3, 7)
LIMIT_BLUR_SIGMA = (0.1, 1.0)

def build_train_transform(crop_size=SIZE_CROP):
    """
    Construct the training-time data augmentation pipeline

    The pipeline includes geometric rotations, brightness/contrast adjustments, Gaussian noise, and blurring to simulate realistic document variations
    """
    return A.Compose([
        A.OneOf([
            A.Rotate(limit=LIMIT_DEG_ROTATE),
            A.Affine(rotate=(-1, 1), shear=(-1, 1)),
        ], p=PROB_AUG_ROTATE),
        A.RandomBrightnessContrast(
            brightness_limit=LIMIT_BRIGHTNESS,
            contrast_limit=LIMIT_CONTRAST,
            p=0.6,
        ),
        A.GaussNoise(std_range=RANGE_STD_GAUSS, p=PROB_AUG_NOISE),
        A.GaussianBlur(blur_limit=LIMIT_BLUR_KERNEL,
                       sigma_limit=LIMIT_BLUR_SIGMA,
                       p=PROB_AUG_BLUR),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])

def build_val_transform(crop_size=SIZE_CROP):
    """
    Construct a deterministic transformation pipeline for validation

    This pipeline only applies normalization and tensor conversion to ensure reproducible evaluation results
    """
    return A.Compose([
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])

class SyntheticLayersDataset(Dataset):
    """
    Multi-label dataset utilizing multi-frame TIFF labels for document layers

    This class manages the ingestion and transformation of synthetic document images and their corresponding masks, supporting filtering for train/test splits
    """

    def __init__(self, path_synthetic_root, transform=None, keep_fn=None):
        """
        Initialize the dataset with synthetic asset paths

        Args:
            path_synthetic_root (str): Root directory containing images/ and labels/
            transform (callable): Data augmentation pipeline
            keep_fn (callable): Optional boolean filter for image paths
        """
        self.path_dir_img = os.path.join(path_synthetic_root, "images")
        self.path_dir_lbl = os.path.join(path_synthetic_root, "labels")

        if not os.path.isdir(self.path_dir_img):
            raise FileNotFoundError(f"images directory missing: {self.path_dir_img}")

        self.image_paths = sorted(glob.glob(os.path.join(self.path_dir_img, "*_input.png")))
        if not self.image_paths:
            self.image_paths = sorted(glob.glob(os.path.join(self.path_dir_img, "*.png")))

        if keep_fn is not None:
            self.image_paths = [p for p in self.image_paths if keep_fn(p)]

        if not self.image_paths:
            raise FileNotFoundError(f"no images found under {self.path_dir_img}")

        self.transform = transform

    def __len__(self):
        """Return the total number of items in the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and transform an image-mask pair

        This method applies background normalization to the RGB image and extracts relevant binary frames from the multi-channel TIFF label
        """
        image_path = self.image_paths[idx]
        image_arr = np.array(Image.open(image_path).convert("RGB"))

        image_arr, _ = normalise_background(image_arr)  # Apply background normalization first

        stem = os.path.basename(image_path).replace("_input.png", "").replace(".png", "")
        label_path = os.path.join(self.path_dir_lbl, f"{stem}_label.tiff")

        if not os.path.exists(label_path):
            mask_arr = np.zeros((image_arr.shape[0], image_arr.shape[1], 3), dtype=np.float32)  # Return empty masks if missing
        else:
            frames = []
            with Image.open(label_path) as lbl_img:
                for i in (IDX_NS, IDX_HW, IDX_PR):
                    lbl_img.seek(i)
                    frames.append(np.array(lbl_img))

            stacked_frames = np.stack(frames, axis=0)
            binary_frames = (stacked_frames > 0).astype(np.float32)  # Binarize masks
            mask_arr = np.transpose(binary_frames, (1, 2, 0))  # Transpose for Albumentations

        if self.transform is None:
            return image_arr, torch.from_numpy(np.transpose(mask_arr, (2, 0, 1)))

        out = self.transform(image=image_arr, mask=mask_arr)

        mask_tensor = out["mask"]
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.permute(2, 0, 1)  # Restore channel-first order for PyTorch

        return out["image"], mask_tensor
