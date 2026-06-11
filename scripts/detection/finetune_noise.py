# -*- coding: utf-8 -*-
"""
Training pipeline for the noise subtype classification model

This module implements the training logic for the ResNet-18 noise classifier. It extracts noise instance crops from synthetic document images using COCO annotations, applies data augmentation, and utilizes class-balanced sampling to address distribution skew. The resulting weights and training history are stored in the standardized output directories
"""

import os
import time
import json
import argparse
import random
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.splits import ensure_split, stems_for, page_stem
from src.detection.noise.classifier import build_classifier, CLASSES_NOISE, SIZE_INPUT

PATH_DIR_SYNTH = os.path.join(BASE_PATH, "data", "synthetic", "factory")
PATH_FILE_SPLIT = os.path.join(PATH_DIR_SYNTH, "split.json")

# Hyperparameter configuration
SIZE_BATCH = 64
COUNT_WORKERS = 4
COUNT_EPOCHS = 30
LR_INIT = 0.01
MOMENTUM_SGD = 0.9
WD_SGD = 1e-4
RATIO_VAL = 0.15
SEED_SPLIT = 42

MAP_CAT_ID = {3: "circles", 4: "lines", 5: "crosses", 6: "marks", 7: "stamps"}  # Map COCO IDs to noise names
NORM_MEAN = (0.485, 0.456, 0.406)  # ImageNet mean statistics
NORM_STD = (0.229, 0.224, 0.225)  # ImageNet standard deviation

def get_artifact_dir(phase: str, model_name: str, artifact_type: str) -> str:
    """
    Retrieve the standardized directory path for model artifacts
    """
    path = os.path.join(BASE_PATH, "outputs", phase, model_name, artifact_type)
    os.makedirs(path, exist_ok=True)
    return path

def _jitter_bbox(x, y, w, h, page_w, page_h, jitter=0.1):
    """
    Apply random spatial expansion to a bounding box
    """
    dx = int(w * jitter * random.uniform(0, 1))
    dy = int(h * jitter * random.uniform(0, 1))
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(page_w, x + w + dx)
    y1 = min(page_h, y + h + dy)
    return x0, y0, x1, y1

def _augment(crop_np):
    """
    Apply stochastic transformations to training image crops
    """
    img = Image.fromarray(crop_np)
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, fillcolor=(245, 242, 235))  # Apply rotation with paper-tone fill
    
    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        new_w = max(SIZE_INPUT, int(img.width * scale))
        new_h = max(SIZE_INPUT, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        left = random.randint(0, max(0, new_w - SIZE_INPUT))
        top = random.randint(0, max(0, new_h - SIZE_INPUT))
        img = img.crop((left, top, left + SIZE_INPUT, top + SIZE_INPUT))
    
    if random.random() < 0.4:
        factor = random.uniform(0.7, 1.3)
        arr = np.array(img).astype(np.float32)
        arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    arr = np.array(img.resize((SIZE_INPUT, SIZE_INPUT), Image.BILINEAR)).astype(np.float32) / 255.0
    mean = np.array(NORM_MEAN, dtype=np.float32)
    std = np.array(NORM_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()

def _val_transform(crop_np):
    """
    Convert image crop into a normalized tensor for validation
    """
    img = Image.fromarray(crop_np).resize((SIZE_INPUT, SIZE_INPUT), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array(NORM_MEAN, dtype=np.float32)
    std = np.array(NORM_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()

class NoiseCropDataset(Dataset):
    """
    Dataset for noise instance crops based on COCO annotations
    """

    def __init__(self, samples, is_train):
        """
        Initialize the noise dataset with sample records
        """
        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        page_img = np.array(Image.open(s["image_path"]).convert("RGB"))
        x, y, w, h = s["bbox"]
        ph, pw = page_img.shape[:2]
        if self.is_train:
            x0, y0, x1, y1 = _jitter_bbox(x, y, w, h, pw, ph)
        else:
            x0, y0, x1, y1 = x, y, x + w, y + h
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(pw, max(x0 + 1, x1)), min(ph, max(y0 + 1, y1))
        crop = page_img[y0:y1, x0:x1]
        if crop.size == 0:
            crop = page_img[:1, :1]  # Guard against degenerate bounding boxes
        tensor = _augment(crop) if self.is_train else _val_transform(crop)
        return tensor, int(s["label_idx"])

def _build_samples(path_synth):
    """
    Parse COCO annotations and generate sample records
    """
    coco_path = os.path.join(path_synth, "coco_instances.json")
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    label_by_name = {name: i for i, name in enumerate(CLASSES_NOISE)}

    samples = []
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in MAP_CAT_ID:
            continue
        class_name = MAP_CAT_ID[cat_id]
        label_idx = label_by_name[class_name]
        img_file = img_by_id.get(ann["image_id"])
        if not img_file:
            continue
        img_path = os.path.join(path_synth, "images", img_file)
        if not os.path.exists(img_path):
            continue
        x, y, w, h = [int(v) for v in ann["bbox"]]
        samples.append({"image_path": img_path, "bbox": [x, y, w, h], "label_idx": label_idx})
    return samples

def main(args):
    """
    Execute the ResNet-18 noise classifier training loop
    """
    out_dir_weights = get_artifact_dir("detection", "noise_resnet", "weights")
    out_dir_logs = get_artifact_dir("detection", "noise_resnet", "logs")
    path_file_best = os.path.join(out_dir_weights, "best.pt")

    random.seed(SEED_SPLIT)

    manifest = ensure_split(os.path.join(args.data, "images"), PATH_FILE_SPLIT)
    train_stems = stems_for(manifest, "train")
    val_stems = stems_for(manifest, "val")
    samples = _build_samples(args.data)
    train_s = [s for s in samples if page_stem(s["image_path"]) in train_stems]
    val_s = [s for s in samples if page_stem(s["image_path"]) in val_stems]
    print(f"[*] Noise crops: train={len(train_s)} val={len(val_s)}")

    class_counts = [0] * len(CLASSES_NOISE)
    for s in train_s:
        class_counts[s["label_idx"]] += 1
    sample_weights = [1.0 / max(1, class_counts[s["label_idx"]]) for s in train_s]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_s), replacement=True)

    ds_train = NoiseCropDataset(train_s, is_train=True)
    ds_val = NoiseCropDataset(val_s, is_train=False)
    loader_train = DataLoader(ds_train, batch_size=args.batch, sampler=sampler,
                               num_workers=COUNT_WORKERS, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                             num_workers=COUNT_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(pretrained=True).to(device)
    opt = SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM_SGD, weight_decay=WD_SGD)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "train_acc": [], "val_acc": [], "lr": [], "epoch_s": []
    }

    best_acc = -1.0
    for epoch in range(args.epochs):
        t_start = time.time()
        model.train()
        loss_sum, n_correct, n_total = 0.0, 0, 0
        
        for imgs, labels in loader_train:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss = crit(outputs, labels)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n_correct += int((outputs.argmax(1) == labels).sum())
            n_total += len(labels)
        
        sched.step()
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, labels in loader_val:
                imgs, labels = imgs.to(device), labels.to(device)
                vc += int((model(imgs).argmax(1) == labels).sum())
                vt += len(labels)
        
        val_acc = vc / max(1, vt)
        train_loss = loss_sum / max(1, n_total / args.batch)
        train_acc = n_correct / max(1, n_total)
        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t_start

        print(f"[{epoch+1:>3}/{args.epochs}] loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f} lr={lr_now:.2e} t={dt:.1f}s")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)
        history["epoch_s"].append(dt)

        with open(os.path.join(out_dir_logs, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_acc": val_acc, "classes": CLASSES_NOISE}, path_file_best)
            print(f"[*] New best val_acc={val_acc:.3f} -> {path_file_best}")

    print(f"[*] Training complete - best val_acc={best_acc:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ResNet-18 noise subtype classifier training")
    p.add_argument("--data", default=PATH_DIR_SYNTH)
    p.add_argument("--epochs", type=int, default=COUNT_EPOCHS)
    p.add_argument("--batch", type=int, default=SIZE_BATCH)
    p.add_argument("--lr", type=float, default=LR_INIT)
    args = p.parse_args()
    main(args)
