"""
train.py — Training DeepLabV3+ for IMX500 deployment
=====================================================
Light training script for Pascal VOC / custom datasets.
Uses CrossEntropyLoss + optional OHEM (Online Hard Example Mining).

Input normalisation: [0, 1]  (NO ImageNet mean/std subtraction)
Reason: IMX500 AI ISP delivers [0,1] data; matching train/deploy
        normalisation is critical for PTQ accuracy.

Usage
-----
  # Pascal VOC 2012
  python train.py \
    --data-root /path/to/VOCdevkit/VOC2012 \
    --dataset voc \
    --epochs 50 \
    --batch-size 8 \
    --num-classes 21 \
    --output-dir ./checkpoints

  # Custom dataset (images/ and masks/ subdirectories)
  python train.py \
    --data-root /path/to/dataset \
    --dataset custom \
    --num-classes 5 \
    --output-dir ./checkpoints
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model


# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

class VOCSegDataset(Dataset):
    """Pascal VOC 2012 semantic segmentation."""

    IGNORE_INDEX = 255

    def __init__(self, root, split="train", img_size=128):
        self.img_size = img_size
        self.split = split

        split_file = os.path.join(
            root, "ImageSets", "Segmentation", f"{split}.txt"
        )
        with open(split_file) as f:
            self.ids = [l.strip() for l in f if l.strip()]

        self.img_dir  = os.path.join(root, "JPEGImages")
        self.mask_dir = os.path.join(root, "SegmentationClass")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img  = Image.open(os.path.join(self.img_dir,  f"{name}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, f"{name}.png"))

        # Resize — BILINEAR for image, NEAREST for mask (no interpolation artefacts)
        img  = img.resize( (self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img_arr  = np.array(img,  dtype=np.float32) / 255.0   # [0,1]
        mask_arr = np.array(mask, dtype=np.int64)

        # VOC uses 255 for border/ignore pixels
        mask_arr[mask_arr == 255] = self.IGNORE_INDEX

        img_tensor  = torch.from_numpy(img_arr.transpose(2, 0, 1))  # CHW
        mask_tensor = torch.from_numpy(mask_arr)
        return img_tensor, mask_tensor


class CustomSegDataset(Dataset):
    """
    Generic dataset layout:
      root/
        images/   *.jpg or *.png
        masks/    *.png  (same stem, integer class indices 0..N-1)
    """

    def __init__(self, root, img_size=128):
        self.img_size = img_size
        img_dir   = os.path.join(root, "images")
        mask_dir  = os.path.join(root, "masks")
        self.pairs = []
        for fname in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(fname)[0]
            mask_path = os.path.join(mask_dir, f"{stem}.png")
            if os.path.exists(mask_path):
                self.pairs.append((
                    os.path.join(img_dir, fname),
                    mask_path,
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = Image.open(img_path).convert("RGB").resize(
                    (self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(mask_path).resize(
                    (self.img_size, self.img_size), Image.NEAREST)

        img_arr  = np.array(img,  dtype=np.float32) / 255.0
        mask_arr = np.array(mask, dtype=np.int64)

        return (torch.from_numpy(img_arr.transpose(2, 0, 1)),
                torch.from_numpy(mask_arr))


# ──────────────────────────────────────────────────────────────────────────────
# Loss: CrossEntropy + optional OHEM
# ──────────────────────────────────────────────────────────────────────────────

class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining cross-entropy.
    Keeps only the top-K hardest pixels per batch for backprop.
    Improves convergence on class-imbalanced segmentation datasets.
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = thresh
        self.min_kept = min_kept
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)   # B×H×W

        valid_mask = targets != self.ignore_index
        loss_valid = loss[valid_mask]

        if loss_valid.numel() == 0:
            return loss.mean()

        # Sort descending and keep top-K hard examples
        sorted_loss, _ = loss_valid.sort(descending=True)
        threshold = max(self.thresh,
                        sorted_loss[min(self.min_kept - 1, len(sorted_loss) - 1)].item())
        hard_mask = loss >= threshold

        final_mask = valid_mask & hard_mask
        if final_mask.sum() == 0:
            return loss_valid.mean()

        return loss[final_mask].mean()


# ──────────────────────────────────────────────────────────────────────────────
# mIoU metric
# ──────────────────────────────────────────────────────────────────────────────

class MeanIoU:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        pred   = pred.cpu().numpy().astype(np.int64)
        target = target.cpu().numpy().astype(np.int64)
        mask = target != self.ignore_index
        pred, target = pred[mask], target[mask]
        idx = target * self.num_classes + pred
        self.confusion += np.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)

    def compute(self):
        iou_list = []
        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            fp = self.confusion[:, c].sum() - tp
            fn = self.confusion[c, :].sum() - tp
            denom = tp + fp + fn
            if denom > 0:
                iou_list.append(tp / denom)
        return np.mean(iou_list) if iou_list else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.dataset == "voc":
        train_ds = VOCSegDataset(args.data_root, "train",  img_size=args.img_size)
        val_ds   = VOCSegDataset(args.data_root, "val",    img_size=args.img_size)
    else:
        train_ds = CustomSegDataset(args.data_root, img_size=args.img_size)
        val_ds   = None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=2) if val_ds else None

    print(f"[Train] Train samples: {len(train_ds)}")
    if val_ds:
        print(f"[Train] Val samples:   {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=args.num_classes,
                        width_mult=args.width_mult).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Parameters: {n_params:,}  "
          f"({n_params * 4 / 1024**2:.2f} MB FP32)")

    # ── Loss & Optimiser ──────────────────────────────────────────────────────
    criterion = OHEMLoss(ignore_index=255) if args.ohem else \
                nn.CrossEntropyLoss(ignore_index=255)

    # Poly learning-rate schedule (standard for segmentation)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    def poly_lr(epoch):
        return (1 - epoch / args.epochs) ** 0.9

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)

    # ── Train ────────────────────────────────────────────────────────────────
    best_miou = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for i, (imgs, masks) in enumerate(train_loader):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 50 == 0:
                avg = total_loss / (i + 1)
                print(f"  Epoch {epoch}/{args.epochs}  "
                      f"Step {i+1}/{len(train_loader)}  "
                      f"Loss: {avg:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

        # ── Validation ────────────────────────────────────────────────────────
        if val_loader and epoch % args.val_every == 0:
            model.eval()
            metric = MeanIoU(args.num_classes)
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs  = imgs.to(device)
                    logits = model(imgs)
                    preds  = logits.argmax(dim=1)
                    metric.update(preds, masks)
            miou = metric.compute()
            print(f"[Epoch {epoch}] Val mIoU: {miou:.4f}")

            if miou > best_miou:
                best_miou = miou
                ckpt_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"[Epoch {epoch}] ✓ Saved best model → {ckpt_path}")

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n[Train] Done. Final model → {final_path}")
    if val_loader:
        print(f"[Train] Best mIoU: {best_miou:.4f}")

    print("\n[Train] Next step: run quantize_and_export.py to prepare for IMX500")
    print(f"  python quantize_and_export.py \\")
    print(f"    --weights {final_path} \\")
    print(f"    --num-classes {args.num_classes} \\")
    print(f"    --calib-dir /path/to/calib/images \\")
    print(f"    --output-dir ./output")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train DeepLabV3+ (MobileNetV2) for IMX500 deployment"
    )
    parser.add_argument("--data-root",   type=str, required=True)
    parser.add_argument("--dataset",     type=str, default="voc",
                        choices=["voc", "custom"])
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--width-mult",  type=float, default=0.35)
    parser.add_argument("--img-size",    type=int, default=128)
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--lr",          type=float, default=0.01)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--ohem",        action="store_true",
                        help="Use Online Hard Example Mining loss")
    parser.add_argument("--val-every",   type=int, default=5)
    parser.add_argument("--output-dir",  type=str, default="./checkpoints")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
