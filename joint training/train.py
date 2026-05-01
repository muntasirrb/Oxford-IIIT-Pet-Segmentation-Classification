import sys
import os
import json
import numpy as np
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

from dataset import get_dataloaders
from model   import UNetJoint
from loss    import JointLoss
from metrics import seg_metrics, cls_metrics


CONFIG = {
    "data_root"     : r"E:\test project\oxford-iiit-pets",
    "image_size"    : 256,
    "batch_size"    : 64,
    "num_workers"   : 4,
    "seg_classes"   : 2,
    "num_breeds"    : 37,
    "features"      : [64, 128, 256, 512],
    "lr"            : 1e-3,
    "num_epochs"    : 100,
    "seg_weight"    : 1.0,
    "cls_weight"    : 1.0,
    "checkpoint_dir": "./checkpoints_joint",
}


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    totals = dict(loss=0, seg_loss=0, cls_loss=0,
                  pixel_acc=0, iou=0, dice=0,
                  top1=0, top5=0, prec=0, rec=0, f1=0)
    n = len(loader)

    for images, masks, breeds, _ in loader:
        images = images.to(device)
        masks  = masks .to(device)
        breeds = breeds.to(device)

        optimizer.zero_grad()
        with autocast("cuda"):
            seg_out, cls_out     = model(images)
            loss, seg_l, cls_l   = criterion(seg_out, cls_out, masks, breeds)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        sm = seg_metrics(seg_out.detach(), masks)
        cm = cls_metrics(cls_out.detach(), breeds)

        totals["loss"]      += loss.item()
        totals["seg_loss"]  += seg_l.item()
        totals["cls_loss"]  += cls_l.item()
        totals["pixel_acc"] += sm["pixel_acc"]
        totals["iou"]       += sm["mean_iou"]
        totals["dice"]      += sm["dice"]
        totals["top1"]      += cm["top1"]
        totals["top5"]      += cm["top5"]
        totals["prec"]      += cm["precision"]
        totals["rec"]       += cm["recall"]
        totals["f1"]        += cm["f1"]

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    totals = dict(loss=0, seg_loss=0, cls_loss=0,
                  pixel_acc=0, iou=0, dice=0,
                  top1=0, top5=0, prec=0, rec=0, f1=0)
    n = len(loader)

    for images, masks, breeds, _ in loader:
        images = images.to(device)
        masks  = masks .to(device)
        breeds = breeds.to(device)

        with autocast("cuda"):
            seg_out, cls_out   = model(images)
            loss, seg_l, cls_l = criterion(seg_out, cls_out, masks, breeds)

        sm = seg_metrics(seg_out, masks)
        cm = cls_metrics(cls_out, breeds)

        totals["loss"]      += loss.item()
        totals["seg_loss"]  += seg_l.item()
        totals["cls_loss"]  += cls_l.item()
        totals["pixel_acc"] += sm["pixel_acc"]
        totals["iou"]       += sm["mean_iou"]
        totals["dice"]      += sm["dice"]
        totals["top1"]      += cm["top1"]
        totals["top5"]      += cm["top5"]
        totals["prec"]      += cm["precision"]
        totals["rec"]       += cm["recall"]
        totals["f1"]        += cm["f1"]

    return {k: v / n for k, v in totals.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    train_loader, val_loader, test_loader, test_ds = get_dataloaders(
        data_root   = CONFIG["data_root"],
        image_size  = CONFIG["image_size"],
        batch_size  = CONFIG["batch_size"],
        num_workers = CONFIG["num_workers"],
    )

    model = UNetJoint(
        in_channels = 3,
        seg_classes = CONFIG["seg_classes"],
        num_breeds  = CONFIG["num_breeds"],
        features    = CONFIG["features"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = JointLoss(
        seg_weight = CONFIG["seg_weight"],
        cls_weight = CONFIG["cls_weight"],
    )
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"])
    scaler    = GradScaler("cuda")

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_iou    = 0.0
    history     = {k: [] for k in [
        "train_loss", "val_loss",
        "train_seg_loss", "val_seg_loss",
        "train_cls_loss", "val_cls_loss",
        "train_pixel_acc", "val_pixel_acc",
        "train_iou", "val_iou",
        "train_dice", "val_dice",
        "train_top1", "val_top1",
        "train_top5", "val_top5",
        "train_prec", "val_prec",
        "train_rec", "val_rec",
        "train_f1", "val_f1",
    ]}

    ckpt_path    = os.path.join(CONFIG["checkpoint_dir"], "best_model_joint.pth")
    history_path = os.path.join(CONFIG["checkpoint_dir"], "history.json")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt["val_iou"]
        print(f"Resumed from epoch {ckpt['epoch']} (best IoU={best_iou:.4f})")
    else:
        print("No checkpoint found, starting from scratch.")

    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        print(f"History loaded ({len(history['train_loss'])} epochs so far)")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CONFIG["num_epochs"] + 1):
        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch [{epoch}/{CONFIG['num_epochs']}]  lr={lr:.6f}")
        print("  Training...", flush=True)

        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device)

        print("  Validating...", flush=True)

        val_m = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Append history
        for k in train_m:
            history[f"train_{k}"].append(train_m[k])
            history[f"val_{k}"]  .append(val_m[k])

        with open(history_path, "w") as f:
            json.dump(history, f)

        # Print
        print(f"  Train  loss={train_m['loss']:.4f}  "
              f"seg={train_m['seg_loss']:.4f}  cls={train_m['cls_loss']:.4f}  "
              f"iou={train_m['iou']:.4f}  dice={train_m['dice']:.4f}  "
              f"acc={train_m['pixel_acc']:.4f}  "
              f"top1={train_m['top1']:.4f}  f1={train_m['f1']:.4f}")
        print(f"  Val    loss={val_m['loss']:.4f}  "
              f"seg={val_m['seg_loss']:.4f}  cls={val_m['cls_loss']:.4f}  "
              f"iou={val_m['iou']:.4f}  dice={val_m['dice']:.4f}  "
              f"acc={val_m['pixel_acc']:.4f}  "
              f"top1={val_m['top1']:.4f}  f1={val_m['f1']:.4f}")

        # Save best based on val IoU
        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_iou"    : val_m["iou"],
                "val_dice"   : val_m["dice"],
                "val_top1"   : val_m["top1"],
                "val_f1"     : val_m["f1"],
            }, ckpt_path)
            print(f"  >> Best model saved "
                  f"(iou={best_iou:.4f}  top1={val_m['top1']:.4f})")

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()