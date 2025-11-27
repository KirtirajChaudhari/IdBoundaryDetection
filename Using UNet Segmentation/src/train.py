#!/usr/bin/env python3
"""
Training script:

- Uses AMP (Automatic Mixed Precision) via torch.cuda.amp for faster & lighter training on GPU.
- Uses a lightweight ResNet-18 encoder by default (when SMP is available).
- Can enable gradient checkpointing to trade compute for memory savings.
"""

import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp  # used for DiceLoss if available

from src.dataset import CardDataset, get_train_transforms, get_valid_transforms
from src.model import build_model


def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0

    use_amp = (device == "cuda")
    for imgs, masks in tqdm(loader, desc="train"):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(imgs)
            loss_bce = F.binary_cross_entropy_with_logits(preds, masks)
            dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
            loss_dice = dice_loss_fn(preds, masks)
            loss = loss_bce + loss_dice

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def valid_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    use_amp = (device == "cuda")

    dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="valid"):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(imgs)
                loss_bce = F.binary_cross_entropy_with_logits(preds, masks)
                loss_dice = dice_loss_fn(preds, masks)
                loss = loss_bce + loss_dice

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train ID-card segmentation model")
    parser.add_argument("--data_root", required=True, help="Root directory with data/images and data/masks")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--checkpoint", required=True, help="Path to save best model state_dict")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet18",
        help="Encoder backbone for SMP U-Net (default: resnet18)",
    )
    parser.add_argument(
        "--no_grad_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (enabled by default on GPU where supported).",
    )
    args = parser.parse_args()

    # Device & performance flags
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    train_imgs = data_root / "images" / "train"
    train_masks = data_root / "masks" / "train"
    val_imgs = data_root / "images" / "val"
    val_masks = data_root / "masks" / "val"

    # Datasets & loaders
    train_ds = CardDataset(train_imgs, train_masks, transforms=get_train_transforms())
    val_ds = CardDataset(val_imgs, val_masks, transforms=get_valid_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Build model with ResNet-18 encoder and (by default) grad checkpointing ON
    use_grad_checkpointing = (not args.no_grad_checkpointing)
    model = build_model(
        preferred="smp",
        encoder=args.encoder,
        encoder_weights="imagenet",
        use_grad_checkpointing=use_grad_checkpointing,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val = float("inf")
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Using encoder: {args.encoder}")
    print(f"Gradient checkpointing: {'ON' if use_grad_checkpointing else 'OFF'}")
    print(f"AMP: {'ON' if device == 'cuda' else 'OFF'}")

    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = valid_epoch(model, val_loader, device)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print("  -> Saved best checkpoint to", ckpt_path)


if __name__ == "__main__":
    main()
