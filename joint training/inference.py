import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import (get_dataloaders, CLASS_NAMES, SPECIES,
                            get_val_transforms)
from model  import UNetJoint


CONFIG = {
    "data_root"      : r"E:\test project\oxford-iiit-pets",
    "image_size"     : 256,
    "batch_size"     : 16,
    "num_workers"    : 4,
    "seg_classes"    : 2,
    "num_breeds"     : 37,
    "features"       : [64, 128, 256, 512],
    "checkpoint_path": "./checkpoints_joint/best_model_joint.pth",
}


def load_model(device):
    model = UNetJoint(
        in_channels = 3,
        seg_classes = CONFIG["seg_classes"],
        num_breeds  = CONFIG["num_breeds"],
        features    = CONFIG["features"],
    ).to(device)
    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def overlay_mask(image_np, mask_np, alpha=0.4):
    """Overlay segmentation mask on image with green = foreground."""
    overlay = image_np.copy()
    # foreground = 0 → green overlay
    overlay[mask_np == 0] = (overlay[mask_np == 0] * (1 - alpha) +
                              np.array([0, 200, 0]) * alpha).astype(np.uint8)
    return overlay


def iou_score(pred_mask, true_mask, cls=0):
    pred    = (pred_mask == cls)
    target  = (true_mask == cls)
    inter   = (pred & target).sum()
    union   = (pred | target).sum()
    return inter / union if union > 0 else 0.0


def run_inference(image_idx=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, _, test_ds = get_dataloaders(
        data_root   = CONFIG["data_root"],
        image_size  = CONFIG["image_size"],
        batch_size  = CONFIG["batch_size"],
        num_workers = CONFIG["num_workers"],
    )

    model = load_model(device)

    # Pick index
    if image_idx is None:
        image_idx = random.randint(0, len(test_ds) - 1)
    print(f"Running inference on test index: {image_idx}")

    image_path   = test_ds.image_paths[image_idx]
    mask_path    = test_ds.mask_paths[image_idx]
    true_breed   = test_ds.breed_labels[image_idx]
    true_species = test_ds.species_labels[image_idx]

    # Load original image for display
    orig_image = np.array(Image.open(image_path).convert("RGB"))

    # Load and process true mask
    true_mask = np.array(Image.open(mask_path)).astype(np.int64)
    true_mask[true_mask == 3] = 1     # boundary → foreground
    true_mask = true_mask - 1         # 1→0 foreground, 2→1 background
    true_mask = np.clip(true_mask, 0, 1)

    # Transform for model
    transforms = get_val_transforms(CONFIG["image_size"])
    aug        = transforms(image=orig_image, mask=true_mask)
    image_t    = aug["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        seg_out, cls_out = model(image_t)

    # Segmentation prediction
    pred_mask = seg_out.argmax(dim=1).squeeze(0).cpu().numpy()

    # Classification prediction
    pred_breed_idx   = cls_out.argmax(dim=1).item()
    pred_breed_name  = CLASS_NAMES[pred_breed_idx]
    true_breed_name  = CLASS_NAMES[true_breed]
    pred_species     = "Cat" if pred_breed_idx in [0,5,6,7,9,11,20,23,26,27,31,32,33] else "Dog"
    true_species_str = SPECIES[true_species]

    # Resize orig for display
    orig_resized = np.array(
        Image.open(image_path).convert("RGB").resize(
            (CONFIG["image_size"], CONFIG["image_size"])))

    # Resize true_mask for display
    true_mask_disp = np.array(
        Image.fromarray(true_mask.astype(np.uint8)).resize(
            (CONFIG["image_size"], CONFIG["image_size"]),
            Image.NEAREST))

    # IoU
    iou = iou_score(pred_mask, true_mask_disp, cls=0)

    # Overlays
    true_overlay = overlay_mask(orig_resized, true_mask_disp)
    pred_overlay = overlay_mask(orig_resized, pred_mask)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    fig.suptitle(
        f'Original class: "{true_breed_name}" ({true_species_str})\n'
        f'Predicted class: "{pred_breed_name}" ({pred_species})\n'
        f'IoU: {iou:.2%}',
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(orig_resized)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(true_overlay)
    axes[1].set_title("Image + True Mask Overlay", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(pred_overlay)
    axes[2].set_title("Image + Predicted Mask Overlay", fontsize=12)
    axes[2].axis("off")

    # Legend
    fg_patch = mpatches.Patch(color=(0, 200/255, 0), label="Foreground (pet)")
    fig.legend(handles=[fg_patch], loc="lower center",
               ncol=1, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig("./checkpoints_joint/inference_result.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved to checkpoints_joint/inference_result.png")


if __name__ == "__main__":
    # Pass any test index or leave empty for random
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None,
                        help="Test set index to run inference on")
    args = parser.parse_args()
    run_inference(image_idx=args.idx)

