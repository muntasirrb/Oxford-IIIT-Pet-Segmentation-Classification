import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import load_all_data, CLASS_NAMES, SPECIES


DATA_ROOT = r"E:\test project\oxford-iiit-pets"


def load_mask(mask_path):
    """Load mask, remap boundary to foreground, return 0-based binary mask."""
    mask = np.array(Image.open(mask_path)).astype(np.int64)
    mask[mask == 3] = 1    # boundary → foreground
    mask = mask - 1        # 1→0 foreground, 2→1 background
    return np.clip(mask, 0, 1)


def overlay_mask(image_np, mask_np, alpha=0.4):
    overlay = image_np.copy().astype(float)
    green   = np.array([0, 200, 0], dtype=float)
    fg      = mask_np == 0
    overlay[fg] = overlay[fg] * (1 - alpha) + green * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_9_random():
    all_images, all_masks, all_breeds, all_species = load_all_data(DATA_ROOT)

    indices = random.sample(range(len(all_images)), 9)

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    fig.suptitle("Dataset Exploration — 9 Random Samples with Mask Overlays",
                 fontsize=15, fontweight="bold")

    for ax, idx in zip(axes.flat, indices):
        image_np = np.array(
            Image.open(all_images[idx]).convert("RGB").resize((256, 256)))
        mask_np  = np.array(
            Image.fromarray(load_mask(all_masks[idx]).astype(np.uint8))
            .resize((256, 256), Image.NEAREST))

        overlaid    = overlay_mask(image_np, mask_np)
        breed_name  = CLASS_NAMES[all_breeds[idx]]
        species_str = SPECIES[all_species[idx]]

        ax.imshow(overlaid)
        ax.set_title(f"{breed_name} ({species_str})", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("./dataset_exploration.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to dataset_exploration.png")


if __name__ == "__main__":
    plot_9_random()