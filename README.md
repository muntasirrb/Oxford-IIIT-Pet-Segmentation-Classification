# Oxford-IIIT Pet Segmentation & Classification

A deep learning pipeline for semantic segmentation and fine-grained breed classification on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) using PyTorch.

---

## Project Overview

This project implements three training pipelines:

| Pipeline | Task | Architecture |
|---|---|---|
| Segmentation | Foreground/background pixel segmentation | U-Net from scratch |
| Classification | 37 breed classification | U-Net encoder + classifier head |
| Joint Training | Both tasks simultaneously | U-Net with dual heads |

---

## Dataset

**Oxford-IIIT Pet Dataset**
- 37 breeds of cats and dogs
- ~200 images per class (~7,349 total)
- Annotations: breed label, species, head bounding box, trimap segmentation mask

**Trimap Classes:**
- Class 1 → Foreground (pet)
- Class 2 → Background
- Class 3 → Border/ambiguous → remapped to foreground

**Stratified Split:**
| Split | Size |
|---|---|
| Train | 5,143 (70%) |
| Val | 1,103 (15%) |
| Test | 1,103 (15%) |

---

## Project Structure

```
Oxford-IIIT-Pet-Segmentation-Classification/
│
├── segmentation/
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── metrics.py
│   ├── train.py
│   ├── test.py
│   └── plot.py
│
├── classification/
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── metrics.py
│   ├── train.py
│   ├── test.py
│   └── plot.py
│
└── joint training/
    ├── dataset.py
    ├── model.py
    ├── loss.py
    ├── metrics.py
    ├── train.py
    ├── test.py
    ├── plot.py
    ├── inference.py
    └── explore.py
```
---

## Model Architecture

### U-Net (Segmentation)

```
Input (3, 256, 256)
    │
    ├── Encoder
    │     ├── Block 1: Conv → BN → ReLU → Conv → BN → ReLU  [64 filters]
    │     ├── Block 2: Conv → BN → ReLU → Conv → BN → ReLU  [128 filters]
    │     ├── Block 3: Conv → BN → ReLU → Conv → BN → ReLU  [256 filters]
    │     └── Block 4: Conv → BN → ReLU → Conv → BN → ReLU  [512 filters]
    │
    ├── Bottleneck: Conv → BN → ReLU → Conv → BN → ReLU     [1024 filters]
    │
    ├── Decoder
    │     ├── Up 1: ConvTranspose + Skip(512) → ConvBlock    [512 filters]
    │     ├── Up 2: ConvTranspose + Skip(256) → ConvBlock    [256 filters]
    │     ├── Up 3: ConvTranspose + Skip(128) → ConvBlock    [128 filters]
    │     └── Up 4: ConvTranspose + Skip(64)  → ConvBlock    [64 filters]
    │
    └── Segmentation Head: Conv 1x1 → (2, 256, 256)
                           0 = Foreground, 1 = Background
```

### U-Net + Classifier Head (Joint)

```
Input (3, 256, 256)
    │
    ├── Encoder (shared)
    │     └── [64 → 128 → 256 → 512]
    │
    ├── Bottleneck (shared)  [1024 filters]
    │         │
    │         ├── Decoder → Segmentation Head → (2, 256, 256)
    │         │
    │         └── Classifier Head
    │               ├── AdaptiveAvgPool → Flatten
    │               ├── Dropout(0.4)
    │               ├── Linear(1024 → 512) → BN → ReLU
    │               ├── Dropout(0.3)
    │               └── Linear(512 → 37) → 37 breed classes
```

---

## Results

### Segmentation (U-Net)
| Metric | Score |
|---|---|
| Pixel Accuracy | 0.94 |
| Mean IoU | 0.90 |
| Dice Score | 0.94 |

### Classification (U-Net Encoder + Head)
| Metric | Score |
|---|---|
| Top-1 Accuracy | 0.59 |
| Precision (Macro) | 0.59 |
| Recall (Macro) | 0.59 |
| F1 Score (Macro) | 0.58 |

### Joint Training
| Metric | Score |
|---|---|
| Pixel Accuracy | 92.72% |
| Mean IoU | 86.06% |
| Dice Score | 92.47% |
| Top-1 Accuracy | 62.65% |
| Precision | 63.13% |
| Recall | 62.66% |
| F1 Score | 61.91% |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/muntasirrb/Oxford-IIIT-Pet-Segmentation-Classification.git
cd Oxford-IIIT-Pet-Segmentation-Classification

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install albumentations scikit-learn matplotlib seaborn numpy pillow tqdm
```

---

## Download Dataset

Download the Oxford-IIIT Pet Dataset from: https://www.robots.ox.ac.uk/~vgg/data/pets/

## Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Scheduler | CosineAnnealingLR |
| Batch Size | 64 |
| Image Size | 256×256 |
| Epochs | 50–100 |
| Mixed Precision | FP16 (torch.amp) |
| Augmentation | Flip, Rotate, ColorJitter, GaussNoise, CoarseDropout |

---

## Hardware

Trained on:
- **GPU:** NVIDIA GeForce RTX 5090 (32GB VRAM)
- **CPU:** AMD RYZEN 9 9950X
- **RAM:** 64GB DDR5

---

## Loss Functions

| Task | Loss |
|---|---|
| Segmentation | Cross-Entropy + Dice Loss |
| Classification | Cross-Entropy with Label Smoothing (0.1) |
| Joint | seg_weight × (CE + Dice) + cls_weight × CE |

---

## References

```bibtex
@InProceedings{parkhi12a,
  author    = {Omkar M. Parkhi and Andrea Vedaldi and
               Andrew Zisserman and C. V. Jawahar},
  title     = {Cats and Dogs},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2012},
}

@inproceedings{ronneberger2015unet,
  author    = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  booktitle = {MICCAI},
  year      = {2015},
}
```

---

## License

This project is for academic and educational purposes.
The Oxford-IIIT Pet Dataset is licensed under
[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
