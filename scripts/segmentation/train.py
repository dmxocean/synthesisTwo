# -*- coding: utf-8 -*-
"""
Training loop for document segmentation models executable workflow

This script implements the training pipeline for SegFormer and UNet models using a single GPU. It handles data loading from synthetic and standardized datasets, performs multi-label loss calculation, and optimizes model parameters using AdamW with a cosine learning rate schedule. Best and last checkpoints are saved based on mean Intersection over Union (mIoU) metrics on a validation split
"""

import os
import time
import json
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from src.core.config import DATA_ROOT, PATH_FILE_SPLIT, get_artifact_dir
from src.core.splits import ensure_split, stems_for, page_stem
from src.segmentation.segformer import build_segformer
from src.segmentation.unet import build_unet
from src.segmentation.data import SyntheticLayersDataset, build_train_transform, build_val_transform
from src.segmentation.loss import MultiLabelLoss, iou_per_class
from src.core.training import EarlyStopping, count_parameters
from src.core.gpu import DeviceManager

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Two independent segmentation models, shared data/loss for a clean A/B
MODELS = {"segformer": build_segformer, "unet": build_unet}

# Tuning parameters
SIZE_BATCH = 32
COUNT_WORKERS = 4
COUNT_EPOCHS = 75
COUNT_EPOCHS_WARM = 5
COUNT_PATIENCE = 10
LR_INIT = 4e-4
WD_INIT = 1e-4
RATIO_VAL = 0.1
SEED_SPLIT = 1337
COUNT_ACCUM_STEPS = 1
KEY_LAYERS = ("ns", "hw", "pr")

PATH_DIR_SYNTH = os.path.join(DATA_ROOT, "synthetic")


def _is_data_root(d):
    """
    Check if a directory contains both images and labels subdirectories
    """
    return os.path.isdir(os.path.join(d, "images")) and os.path.isdir(os.path.join(d, "labels"))


def _find_data_roots(path_data):
    """
    Search recursively for directories containing paired images and labels
    """
    roots = []
    if _is_data_root(path_data):
        roots.append(path_data)

    for item in sorted(os.listdir(path_data)):
        sub = os.path.join(path_data, item)
        if not os.path.isdir(sub):
            continue
        if _is_data_root(sub):
            roots.append(sub)
        else:
            for subitem in sorted(os.listdir(sub)):
                deep_sub = os.path.join(sub, subitem)
                if os.path.isdir(deep_sub) and _is_data_root(deep_sub):
                    roots.append(deep_sub)
    return roots


def _build_loaders(data_roots, size_batch, count_workers):
    """
    Construct training and validation data loaders from identified data roots
    """
    img_dirs = [os.path.join(r, "images") for r in data_roots]
    manifest = ensure_split(img_dirs, PATH_FILE_SPLIT)
    train_stems = stems_for(manifest, "train")
    val_stems = stems_for(manifest, "val")

    train_ds = ConcatDataset([SyntheticLayersDataset(r, build_train_transform(),
                              keep_fn=lambda p: page_stem(p) in train_stems) for r in data_roots])
    val_ds = ConcatDataset([SyntheticLayersDataset(r, build_val_transform(),
                              keep_fn=lambda p: page_stem(p) in val_stems) for r in data_roots])

    loader_train = DataLoader(train_ds, batch_size=size_batch, shuffle=True,
                              num_workers=count_workers, pin_memory=True, drop_last=True)
    loader_val = DataLoader(val_ds, batch_size=size_batch, shuffle=False,
                              num_workers=count_workers, pin_memory=True)

    print(f"Split {os.path.basename(PATH_FILE_SPLIT)}: "
          + " ".join(f"{k}={v}" for k, v in manifest["counts"].items())
          + f" pages | train={len(train_ds)} val={len(val_ds)} tiles (test held out)", flush=True)
    return loader_train, loader_val


def _validate(model, loader, device):
    """
    Calculate mean and per-class IoU on the validation dataset
    """
    model.eval()
    iou_acc = torch.zeros(len(KEY_LAYERS), device=device)
    n_batch = 0
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(img)
            iou_acc += iou_per_class(logits.float(), mask, num_classes=len(KEY_LAYERS))
            n_batch += 1

    iou_per = (iou_acc / max(1, n_batch)).cpu().tolist()
    return float(sum(iou_per) / len(iou_per)), iou_per


def _save_checkpoint(path, model, optimizer, epoch, best_iou, per_class):
    """
    Save model state and training metadata to a checkpoint file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    ckpt = {
        "model_state":     state_dict,
        "optimizer_state": optimizer.state_dict(),
        "epoch":           epoch,
        "best_iou_mean":   best_iou,
        "iou_per_class":   per_class,
        "layer_keys":      list(KEY_LAYERS),
    }
    torch.save(ckpt, path)


def main(args):
    """
    Execution logic for the segmentation model training loop
    """
    device = DeviceManager.get_device()
    DeviceManager.print_hardware_summary()

    model_ckpt_dir = get_artifact_dir("segmentation", args.model, "weights")
    model_logs_dir = get_artifact_dir("segmentation", args.model, "logs")

    run_name = f"run_{int(time.time())}"
    path_best = os.path.join(model_ckpt_dir, "best.pt")
    path_last = os.path.join(model_ckpt_dir, "last.pt")
    path_tb = os.path.join(model_logs_dir, "tb", run_name)
    writer = SummaryWriter(log_dir=path_tb)

    print(f"Device: {device}", flush=True)
    print(f"Checkpoints: {model_ckpt_dir} (best.pt + last.pt)", flush=True)
    print(f"TensorBoard: tensorboard --logdir {model_logs_dir}", flush=True)

    data_roots = _find_data_roots(args.data)
    print(f"Data roots found: {data_roots}", flush=True)

    # Adjust batch size and accumulation per model if using defaults
    if args.batch == SIZE_BATCH:
        if args.model == "segformer":
            args.batch = 12
            args.accum = 1
        elif args.model == "unet":
            args.batch = 48
            args.accum = 1

    print(f"Train {args.model} | data={args.data} epochs={args.epochs} batch={args.batch}", flush=True)

    history = {"train_loss": [], "mean_iou": [],
               "iou_hw": [], "iou_pr": [], "iou_ns": [],
               "lr": [], "epoch_s": []}

    loader_train, loader_val = _build_loaders(data_roots, args.batch, args.workers)

    model = MODELS[args.model](pretrained=True).to(device)

    trainable, total = count_parameters(model)
    print(f"Model params: {trainable:,} trainable / {total:,} total", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WD_INIT)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - COUNT_EPOCHS_WARM))
    scaler = torch.amp.GradScaler('cuda')
    early_stop = EarlyStopping(patience=args.patience, mode='max')

    criterion = MultiLabelLoss().to(device)

    start_epoch = 0
    best_mean_iou = -1.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_mean_iou = float(ckpt.get("best_iou_mean", -1.0))
        early_stop.best_score = best_mean_iou
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if epoch < COUNT_EPOCHS_WARM:
            lr_now = args.lr * (epoch + 1) / COUNT_EPOCHS_WARM
            for g in optimizer.param_groups:
                g["lr"] = lr_now

        model.train()
        loss_sum = 0.0
        n_batch = 0
        t_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        for i, (img, mask) in enumerate(loader_train):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(img)
                loss = criterion(logits, mask) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.item()) * args.accum
            n_batch += 1

        if epoch >= COUNT_EPOCHS_WARM:
            scheduler.step()

        mean_iou, per_class = _validate(model, loader_val, device)
        train_loss = loss_sum / max(1, n_batch)

        dt = time.time() - t_start
        lr_now = optimizer.param_groups[0]["lr"]
        per_class_str = " ".join(f"{k}={v:.3f}" for k, v in zip(KEY_LAYERS, per_class))
        print(f"Epoch {epoch+1}/{args.epochs} loss={train_loss:.4f} "
              f"mIoU={mean_iou:.4f} {per_class_str} "
              f"lr={lr_now:.2e} t={dt:.1f}s", flush=True)

        writer.add_scalar("train/loss",    train_loss, epoch)
        writer.add_scalar("val/mIoU_mean", mean_iou,   epoch)
        for k, v in zip(KEY_LAYERS, per_class):
            writer.add_scalar(f"val/iou_{k}", v, epoch)
        writer.add_scalar("lr",           lr_now, epoch)
        writer.add_scalar("time/epoch_s", dt,     epoch)
        writer.flush()

        history["train_loss"].append(train_loss)
        history["mean_iou"].append(mean_iou)
        for k, v in zip(KEY_LAYERS, per_class):
            history[f"iou_{k}"].append(v)
        history["lr"].append(lr_now)
        history["epoch_s"].append(dt)

        with open(os.path.join(model_logs_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        _save_checkpoint(path_last, model, optimizer, epoch, mean_iou, per_class)
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            _save_checkpoint(path_best, model, optimizer, epoch, mean_iou, per_class)
            print(f"New best mIoU={mean_iou:.4f} saved to {path_best}", flush=True)

        if early_stop(mean_iou):
            print(f"Early stopping triggered at epoch {epoch+1} (no improvement for {args.patience} epochs)", flush=True)
            break

    print(f"Training complete | best mIoU={best_mean_iou:.4f}", flush=True)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegFormer for 3-class document segmentation")
    parser.add_argument("--model",     type=str,   default="segformer", choices=list(MODELS))
    parser.add_argument("--data",      type=str,   default=PATH_DIR_SYNTH)
    parser.add_argument("--epochs",    type=int,   default=COUNT_EPOCHS)
    parser.add_argument("--patience",  type=int,   default=COUNT_PATIENCE)
    parser.add_argument("--batch",     type=int,   default=SIZE_BATCH)
    parser.add_argument("--lr",        type=float, default=LR_INIT)
    parser.add_argument("--workers",   type=int,   default=COUNT_WORKERS)
    parser.add_argument("--accum",     type=int,   default=COUNT_ACCUM_STEPS)
    parser.add_argument("--resume",    type=str,   default="")
    args = parser.parse_args()
    main(args)
