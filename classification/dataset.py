import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


CLASS_NAMES = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier",
    "Basset Hound", "Beagle", "Bengal", "Birman", "Bombay", "Boxer",
    "British Shorthair", "Chihuahua", "Egyptian Mau", "English Cocker Spaniel",
    "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese",
    "Japanese Chin", "Keeshond", "Leonberger", "Maine Coon",
    "Miniature Pinscher", "Newfoundland", "Persian", "Pomeranian", "Pug",
    "Ragdoll", "Russian Blue", "Saint Bernard", "Samoyed",
    "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx",
    "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier",
]

SPECIES = {0: "Cat", 1: "Dog"}


def get_train_transforms(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.3),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                 rotate=(-15, 15), p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(num_holes_range=(1, 8),
                        hole_height_range=(16, 32),
                        hole_width_range=(16, 32),
                        fill=0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_all_data(data_root):
    """
    Load all images and labels from both trainval and test splits.
    Remaps boundary pixels (3) to foreground (1) as per requirement.
    Returns image paths, mask paths, breed labels, species labels.
    """
    img_dir  = os.path.join(data_root, "images")
    ann_dir  = os.path.join(data_root, "annotations")
    trim_dir = os.path.join(ann_dir, "trimaps")

    all_images  = []
    all_masks   = []
    all_breeds  = []
    all_species = []

    for split in ["trainval", "test"]:
        list_file = os.path.join(ann_dir, f"{split}.txt")
        with open(list_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                name       = parts[0]
                class_id   = int(parts[1]) - 1   # breed: 0-36
                species_id = int(parts[2]) - 1   # species: 0=cat, 1=dog
                img_path   = os.path.join(img_dir,  f"{name}.jpg")
                mask_path  = os.path.join(trim_dir, f"{name}.png")
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    all_images .append(img_path)
                    all_masks  .append(mask_path)
                    all_breeds .append(class_id)
                    all_species.append(species_id)

    print(f"[Dataset] Total images found: {len(all_images)}")
    return all_images, all_masks, all_breeds, all_species


def stratified_split(all_images, all_masks, all_breeds, all_species,
                     val_split=0.15, test_split=0.15, seed=42):
    """Stratified split ensuring equal class representation."""

    # Split off test first
    (tv_imgs, test_imgs,
     tv_masks, test_masks,
     tv_breeds, test_breeds,
     tv_species, test_species) = train_test_split(
        all_images, all_masks, all_breeds, all_species,
        test_size    = test_split,
        stratify     = all_breeds,
        random_state = seed,
    )

    # Split remaining into train and val
    val_ratio = val_split / (1 - test_split)
    (train_imgs, val_imgs,
     train_masks, val_masks,
     train_breeds, val_breeds,
     train_species, val_species) = train_test_split(
        tv_imgs, tv_masks, tv_breeds, tv_species,
        test_size    = val_ratio,
        stratify     = tv_breeds,
        random_state = seed,
    )

    print(f"[Dataset] Train: {len(train_imgs)} | "
          f"Val: {len(val_imgs)} | "
          f"Test: {len(test_imgs)}")

    return (train_imgs,  train_masks,  train_breeds,  train_species,
            val_imgs,    val_masks,    val_breeds,    val_species,
            test_imgs,   test_masks,   test_breeds,   test_species)


class OxfordPetJointDataset(Dataset):
    """
    Joint dataset for segmentation + classification.

    Trimap values:
        1 → foreground (pet)
        2 → background
        3 → boundary  → remapped to foreground (1) as per requirement

    Final mask classes (0-based):
        0 → foreground
        1 → background
    """
    def __init__(self, image_paths, mask_paths, breed_labels,
                 species_labels, transforms=None):
        self.image_paths   = image_paths
        self.mask_paths    = mask_paths
        self.breed_labels  = breed_labels
        self.species_labels= species_labels
        self.transforms    = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask  = np.array(Image.open(self.mask_paths[idx])).astype(np.int64)

        # Remap boundary pixels (3) → foreground (1)
        # Then convert to 0-based: foreground=0, background=1
        mask[mask == 3] = 1   # boundary → foreground
        mask = mask - 1       # 1→0 (foreground), 2→1 (background)
        mask = np.clip(mask, 0, 1)

        breed   = self.breed_labels[idx]
        species = self.species_labels[idx]

        if self.transforms:
            out   = self.transforms(image=image, mask=mask)
            image = out["image"]
            mask  = out["mask"].long()

        return (image,
                mask,
                torch.tensor(breed,   dtype=torch.long),
                torch.tensor(species, dtype=torch.long))


def get_dataloaders(data_root, image_size=256, batch_size=16,
                    num_workers=4, val_split=0.15, test_split=0.15, seed=42):

    all_images, all_masks, all_breeds, all_species = load_all_data(data_root)

    (train_imgs,  train_masks,  train_breeds,  train_species,
     val_imgs,    val_masks,    val_breeds,    val_species,
     test_imgs,   test_masks,   test_breeds,   test_species) = stratified_split(
        all_images, all_masks, all_breeds, all_species,
        val_split=val_split, test_split=test_split, seed=seed,
    )

    train_ds = OxfordPetJointDataset(
        train_imgs, train_masks, train_breeds, train_species,
        transforms=get_train_transforms(image_size))

    val_ds = OxfordPetJointDataset(
        val_imgs, val_masks, val_breeds, val_species,
        transforms=get_val_transforms(image_size))

    test_ds = OxfordPetJointDataset(
        test_imgs, test_masks, test_breeds, test_species,
        transforms=get_val_transforms(image_size))

    # Store test data for inference script
    test_ds.image_paths    = test_imgs
    test_ds.mask_paths     = test_masks
    test_ds.breed_labels   = test_breeds
    test_ds.species_labels = test_species

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, test_ds