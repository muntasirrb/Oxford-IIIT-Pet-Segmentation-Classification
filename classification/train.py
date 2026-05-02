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
from metrics import cls_metrics


CONFIG = {
    "data_root"     : r"E:\test project\oxford-iiit-pets",
    "image_size"    : 256,
    "batch_size"    : 192,
    "num_workers"   : 8,
    "seg_classes"   : 2,
    "num_breeds"    : 37,
    "features"      : [64, 128, 256, 512],
    "lr"            : 1e-3,
    "num_epochs"    : 100,
    "seg_weight"    : 0.0,
    "cls_weight"    : 1.0,
    "checkpoint_dir": "./checkpoints_cls_only",
}


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    totals = dict(loss=0, cls_loss=0,
                  top1=0, top5=0,
                  prec=0, rec=0, f1=0)
    n = len(loader)

    for images, masks, breeds, _ in loader:
        images = images.to(device)
        breeds = breeds.to(device)

        optimizer.zero_grad()
        with autocast("cuda"):
            _, cls_out               = model(images, seg=False)
            loss, seg_loss, cls_loss = criterion(None, cls_out, None, breeds)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        cm = cls_metrics(cls_out.detach(), breeds)

        totals["loss"]     += loss.item()
        totals["cls_loss"] += cls_loss.item()
        totals["top1"]     += cm["top1"]
        totals["top5"]     += cm["top5"]
        totals["prec"]     += cm["precision"]
        totals["rec"]      += cm["recall"]
        totals["f1"]       += cm["f1"]

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    totals = dict(loss=0, cls_loss=0,
                  top1=0, top5=0,
                  prec=0, rec=0, f1=0)
    n = len(loader)

    for images, masks, breeds, _ in loader:
        images = images.to(device)
        breeds = breeds.to(device)

        with autocast("cuda"):
            _, cls_out               = model(images, seg=False)
            loss, seg_loss, cls_loss = criterion(None, cls_out, None, breeds)

        cm = cls_metrics(cls_out, breeds)

        totals["loss"]     += loss.item()
        totals["cls_loss"] += cls_loss.item()
        totals["top1"]     += cm["top1"]
        totals["top5"]     += cm["top5"]
        totals["prec"]     += cm["precision"]
        totals["rec"]      += cm["recall"]
        totals["f1"]       += cm["f1"]

    return {k: v / n for k, v in totals.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # Data
    train_loader, val_loader, test_loader, test_ds = get_dataloaders(
        data_root   = CONFIG["data_root"],
        image_size  = CONFIG["image_size"],
        batch_size  = CONFIG["batch_size"],
        num_workers = CONFIG["num_workers"],
    )

    # Model
    model = UNetJoint(
        in_channels = 3,
        seg_classes = CONFIG["seg_classes"],
        num_breeds  = CONFIG["num_breeds"],
        features    = CONFIG["features"],
    ).to(device)

    # Freeze decoder and seg head
    for param in model.ups.parameters():
        param.requires_grad = False
    for param in model.dec_blocks.parameters():
        param.requires_grad = False
    for param in model.seg_head.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Loss, optimizer, scheduler, scaler
    criterion = JointLoss(
        seg_weight = CONFIG["seg_weight"],
        cls_weight = CONFIG["cls_weight"],
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"], weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"])
    scaler    = GradScaler("cuda")

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_top1   = 0.0
    history     = {
        "train_loss"    : [], "val_loss"    : [],
        "train_cls_loss": [], "val_cls_loss": [],
        "train_top1"    : [], "val_top1"    : [],
        "train_top5"    : [], "val_top5"    : [],
        "train_prec"    : [], "val_prec"    : [],
        "train_rec"     : [], "val_rec"     : [],
        "train_f1"      : [], "val_f1"      : [],
    }

    ckpt_path    = os.path.join(CONFIG["checkpoint_dir"], "best_model_cls.pth")
    history_path = os.path.join(CONFIG["checkpoint_dir"], "history.json")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        best_top1   = ckpt["val_top1"]
        print(f"Resumed from epoch {ckpt['epoch']} "
              f"(best top1={best_top1:.4f}  f1={ckpt['val_f1']:.4f})")
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
        history["train_loss"]    .append(train_m["loss"])
        history["val_loss"]      .append(val_m["loss"])
        history["train_cls_loss"].append(train_m["cls_loss"])
        history["val_cls_loss"]  .append(val_m["cls_loss"])
        history["train_top1"]    .append(train_m["top1"])
        history["val_top1"]      .append(val_m["top1"])
        history["train_top5"]    .append(train_m["top5"])
        history["val_top5"]      .append(val_m["top5"])
        history["train_prec"]    .append(train_m["prec"])
        history["val_prec"]      .append(val_m["prec"])
        history["train_rec"]     .append(train_m["rec"])
        history["val_rec"]       .append(val_m["rec"])
        history["train_f1"]      .append(train_m["f1"])
        history["val_f1"]        .append(val_m["f1"])

        # Save history every epoch
        with open(history_path, "w") as f:
            json.dump(history, f)

        # Print
        print(f"  Train  loss={train_m['loss']:.4f}  "
              f"cls={train_m['cls_loss']:.4f}  "
              f"top1={train_m['top1']:.4f}  "
              f"top5={train_m['top5']:.4f}  "
              f"prec={train_m['prec']:.4f}  "
              f"rec={train_m['rec']:.4f}  "
              f"f1={train_m['f1']:.4f}")
        print(f"  Val    loss={val_m['loss']:.4f}  "
              f"cls={val_m['cls_loss']:.4f}  "
              f"top1={val_m['top1']:.4f}  "
              f"top5={val_m['top5']:.4f}  "
              f"prec={val_m['prec']:.4f}  "
              f"rec={val_m['rec']:.4f}  "
              f"f1={val_m['f1']:.4f}")

        # Save best checkpoint
        if val_m["top1"] > best_top1:
            best_top1 = val_m["top1"]
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_top1"   : val_m["top1"],
                "val_top5"   : val_m["top5"],
                "val_loss"   : val_m["loss"],
                "val_f1"     : val_m["f1"],
            }, ckpt_path)
            print(f"  >> Best model saved "
                  f"(top1={best_top1:.4f}  f1={val_m['f1']:.4f})")

    print(f"\nTraining complete. Best val Top-1: {best_top1:.4f}")


if __name__ == "__main__":
    main()