# 🪪 RefurbEdge — ID Card Segmentation & Rectification

An end-to-end, non-YOLO pipeline for detecting, segmenting, estimating corners, and rectifying ID cards from unconstrained images. This project implements a robust U-Net + Geometry approach, focusing on explainability, practicality, and minimal dependencies.

---

## 🌟 Project Overview

This pipeline was developed as an AI Trainee (Computer Vision) assignment with a core constraint: **avoid complex black-box models like YOLO, SAM, and heavy OpenCV heuristic pipelines**.

### High-Level Approach

The core process is intentionally simple and interview-friendly:

1. **Semantic Segmentation**: A U-Net (with a ResNet-18 encoder) is trained to produce a binary mask distinguishing the "card" from the "background."

2. **Geometry Extraction**: The mask is processed to find the largest polygon. The Minimum Rotated Rectangle of this polygon is computed, and its four vertices are used as the estimated card corners.

3. **Rectification**: A projective transform is applied using the four corners to warp the card into a canonical, top-down view (e.g., 600×400 px).

**The Sell**: "We segment the card region first, then we use simple geometric principles (minimum bounding box) on that mask to locate the four corners and rectify the image."

---

## 🚀 Quick Start (Windows PowerShell)

The following steps assume you have Python 3.10+ and have updated the paths in `scripts/merge_config.json` to point to your local dataset directories.

### 1. Setup Environment

```powershell
# Create and activate virtualenv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# (Optional for Roboflow COCO exports on Windows)
# pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

### 2. Merge Datasets

This script unifies various formats (pre-generated masks, MIDV-500 JSON quadrilaterals, COCO) into the standard `data/` structure.

```powershell
# Merge into .\data
python .\scripts\merge_datasets.py --config .\scripts\merge_config.json --out .\data --seed 42
```

### 3. Training

It's recommended to start with the smaller subset for faster experimentation on limited VRAM (e.g., GTX 1650).

#### A. Train on Small Subset (Recommended)

```powershell
# 3.1 Create a smaller subset for faster experiments
python .\scripts\make_small_subset.py `
    --manifest .\data\manifest.csv `
    --out .\data_small `
    --max_per_dataset 2000

# 3.2 Train the model
python -m src.train `
    --data_root .\data_small `
    --checkpoint .\checkpoints\unet_idcard_small.pth `
    --epochs 5 `
    --batch 4 `
    --lr 1e-4
```

#### B. Train on Full Dataset

```powershell
python -m src.train `
    --data_root .\data `
    --checkpoint .\checkpoints\unet_idcard.pth `
    --epochs 3 `
    --batch 4 `
    --lr 1e-4
```

### 4. Inference

Run the pipeline on a single image to generate the mask, overlay, corners, and rectified crop.

```powershell
python -m src.infer `
    .\sample_outputs\a1.jpeg `
    .\checkpoints\unet_idcard.pth `
    .\sample_outputs\inference_outputs\test1
```

### 5. Evaluation

Compute standard and geometry-aware metrics (IoU, Dice, NCD) against the ground-truth test masks.

```powershell
python -m src.evaluate `
    --pred_dir .\out `
    --gt_dir .\data\masks\test `
    --out .\out\report.csv
```

---

## 📂 Repository Structure

```
RefurbEdge/
├── README.md
├── requirements.txt
│
├── docs/
│   └── project_requirements.md    # Detailed Functional/Non-Functional Specs
│
├── scripts/                       # Data preparation and QA tools
│   ├── merge_datasets.py
│   ├── merge_config.json          # Config for original dataset paths
│   └── make_small_subset.py
│
└── src/                           # Core pipeline logic
    ├── dataset.py                 # CardDataset + Albumentations
    ├── model.py                   # U-Net/ResNet-18 factory
    ├── train.py                   # Training loop (AMP, Grad Checkpointing)
    ├── infer.py                   # Segmentation, Geometry, and Warping
    ├── evaluate.py                # IoU, Dice, NCD metrics
    └── utils.py                   # Polygon/Warp utilities
```

---

## 🛠 Tech Stack & Design Choices

| Component | Framework / Library | Rationale |
|-----------|-------------------|-----------|
| **Deep Learning** | PyTorch, segmentation_models_pytorch | Industry standard, robust U-Net implementation. |
| **Model** | U-Net with ResNet-18 Encoder | Simple, fast to train, excellent balance of performance/explainability. |
| **Acceleration** | torch.amp.autocast, Grad Checkpointing | Crucial for GTX 1650 (4GB VRAM). Reduces memory footprint and speeds up training. |
| **Data Augmentation** | Albumentations | Rich, fast, GPU-friendly augmentations for rotation, lighting, and perspective. |
| **Geometry** | Shapely, scikit-image | Reliable, simple libraries for polygon operations (largest polygon, minimum rotated rectangle). |
| **Metrics** | IoU, Dice, NCD (Normalized Corner Distance) | Includes a geometry-aware metric to evaluate the final corner estimation quality. |

---

## 📋 Full Project Requirements

A detailed breakdown of the functional, non-functional, and technical requirements is available in `docs/project_requirements.md`.

### Key Functional Requirements

- **Input/Output**: Accepts an image, outputs a binary mask, a JSON of four corners, and a 600×400 rectified card crop.
- **Dataset Handling**: Support for merging MIDV-2020/500, SmartDoc, Roboflow (COCO), and custom images.
- **Evaluation**: Compute and report IoU, Dice, and NCD (Normalized Corner Distance).

### Key Non-Functional Requirements

- **Explainability**: Reliance on explicit Segmentation → Geometry path.
- **Performance**: Must run efficiently on a single GPU (e.g., GTX 1650), leveraging AMP and gradient checkpointing.
- **Robustness**: Gracefully handles corrupted files and missing ground-truth masks without crashing.