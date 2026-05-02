import sys
import os
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.amp import autocast
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from dataset import get_dataloaders
from model   import UNetJoint
from loss    import JointLoss
from metrics import cls_metrics


CONFIG = {
    "data_root"      : r"E:\test project\oxford-iiit-pets",  # ← fix typo (pets → pet)
    "image_size"     : 256,
    "batch_size"     : 64,
    "num_workers"    : 8,
    "seg_classes"    : 2,
    "num_breeds"     : 37,
    "features"       : [64, 128, 256, 512],
    "checkpoint_path": "./checkpoints_cls_only/best_model_cls.pth",
    "checkpoint_dir" : "./checkpoints_cls_only",
}


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, _ = get_dataloaders(
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

    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val top1={ckpt['val_top1']:.4f}  f1={ckpt['val_f1']:.4f})")

    criterion = JointLoss(
        seg_weight = 0.0,
        cls_weight = 1.0,
    )
    model.eval()

    total_loss  = 0.0
    all_preds   = []
    all_targets = []
    n           = len(test_loader)

    with torch.no_grad():
        for i, (images, masks, breeds, _) in enumerate(test_loader):
            images = images.to(device)
            breeds = breeds.to(device)

            with autocast("cuda"):
                _, cls_out               = model(images, seg=False)
                loss, _, cls_loss        = criterion(None, cls_out, None, breeds)

            cm = cls_metrics(cls_out, breeds)

            total_loss  += loss.item()
            all_preds   .extend(cm["preds"])
            all_targets .extend(cm["targets"])

            print(f"  [{i+1}/{n}]  loss={total_loss/(i+1):.4f}  "
                  f"top1={cm['top1']:.4f}  "
                  f"f1={cm['f1']:.4f}", flush=True)

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    cm_matrix   = confusion_matrix(all_targets, all_preds)

    results = {
        "test_loss"       : total_loss / n,
        "test_top1"       : float((all_preds == all_targets).mean()),
        "test_precision"  : float(precision_score(all_targets, all_preds,
                                                  average="macro", zero_division=0)),
        "test_recall"     : float(recall_score(all_targets, all_preds,
                                               average="macro", zero_division=0)),
        "test_f1"         : float(f1_score(all_targets, all_preds,
                                           average="macro", zero_division=0)),
        "confusion_matrix": cm_matrix.tolist(),
    }

    results_path = os.path.join(CONFIG["checkpoint_dir"], "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n== Test Results (Classification Only) ==")
    print(f"  Loss      : {results['test_loss']:.4f}")
    print(f"  Top-1 Acc : {results['test_top1']:.4f}")
    print(f"  Precision : {results['test_precision']:.4f}")
    print(f"  Recall    : {results['test_recall']:.4f}")
    print(f"  F1 Score  : {results['test_f1']:.4f}")
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    test()