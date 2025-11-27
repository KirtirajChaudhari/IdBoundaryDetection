# 🪪 RefurbEdge — ID Card Segmentation & Rectification

### *Two-Approach Comparative Solution: U-Net Segmentation + Geometry & DeepLabV3+ Semantic Segmentation*

This repository presents two independent and explainable approaches to **ID card detection, segmentation, corner estimation, and rectification**, designed under the requirement:

> **No YOLO, no SAM, no OpenCV-only heuristics.**
> Focus on **simple, practical, reproducible ML** suitable for interview & real-world use.

These models operate across uncontrolled image scenarios:

* Varied lighting & backgrounds
* Arbitrary orientations & device-captured images
* Multiplicity of document types (IDs, passports, licenses)

---

## 🌟 Overview of Both Approaches

| Approach                                          | Folder                    | Core Technique                                                 | Key Output                        | Strength                                 |
| ------------------------------------------------- | ------------------------- | -------------------------------------------------------------- | --------------------------------- | ---------------------------------------- |
| **Approach A — U-Net Segmentation + Geometry**    | `Using UNet Segmentation` | U-Net (ResNet-18 encoder) + Shapely minimum bounding rectangle | Mask + 4 Corners + Rectified Crop | Best for corner accuracy & rectification |
| **Approach B — DeepLabV3+ Semantic Segmentation** | `Using DeepLabV3+`        | DeepLabV3+ (ResNet34) segmentation                             | Mask + Overlay + Boundary outline | Strong mask quality & cleaner edges      |

---

## 🎯 Problem Statement

Given an image possibly containing an ID card:

1. Segment the ID card region.
2. Derive boundary geometry / corner points.
3. Produce a warped fronto-parallel view (Approach A).
4. Generate boundary outline (Approach B).

Both solutions support:

* **GPU acceleration**
* **Mixed precision (AMP)**
* **Dataset merging + augmentation**
* **Metrics: IoU, Dice, NCD**

---

## 📁 Repository Structure

```
IdBoundaryDetection/
│
├── README.md                               <-- THIS FILE
├── requirements.txt
├── .gitignore
├── .gitattributes                          <-- For Git LFS model management
│
├── Using UNet Segmentation/                <-- Approach A (Primary solution)
│   ├── src/
│   ├── scripts/
│   ├── sample_outputs/
│   ├── checkpoints/unet_idcard_best.pth    <-- via Git LFS
│   └── README.md                           <-- Detailed documentation
│
├── Using DeepLabV3+/                       <-- Approach B (Alternative solution)
│   ├── infer_and_visualize.py
│   ├── deeplabv3p_best.pth                 <-- via Git LFS
│   └── README.md
│
└── LICENSE
```

---

## 🧠 Approach A — U-Net Segmentation + Geometry (Primary)

### Pipeline

1. **Binary segmentation (foreground card / background)**
2. **Polygon extraction → Largest connected region**
3. **Minimum rotated rectangle → 4 corners**
4. **Projective warp → Rectified standardized crop**

### Why this works

> Segment first → Geometry second
> More stable than regression-based corner prediction, and more explainable than YOLO-style detectors.

### Example Command

```powershell
python -m src.infer .\sample_outputs\a1.jpeg .\checkpoints\unet_idcard_best.pth .\sample_outputs\inference_outputs\test1
```

Output includes:

```
mask.png
overlay.png
corners.json
warp.png
```

---

## 🧠 Approach B — DeepLabV3+ Semantic Segmentation (Alternative)

### Motivation

Better boundary smoothness and general segmentation robustness vs. U-Net on noisy datasets.

### Output

Mask + overlay + **precise ID card boundary outline**

### Example Command

```bash
python infer_and_visualize.py --image samples/input.jpg --output-dir results
```

---

## 🚀 Setup & Training Instructions

### Create environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Merge datasets for Approach A

```powershell
python .\scripts\merge_datasets.py --config .\scripts\merge_config.json --out .\data --seed 42
```

### Train U-Net (recommended subset first)

```powershell
python -m src.train --data_root .\data_small --checkpoint .\checkpoints\unet_idcard_best.pth --epochs 5 --batch 4 --lr 1e-4
```

### Evaluate

```powershell
python -m src.evaluate --pred_dir .\out --gt_dir .\data\masks\test --out .\out\report.csv
```

---

## 🛠 Tech Stack & Tools

| Category          | Tools                                 |
| ----------------- | ------------------------------------- |
| Deep Learning     | PyTorch, AMP, Grad Checkpointing      |
| Segmentation      | U-Net, DeepLabV3+                     |
| Data Augmentation | Albumentations                        |
| Geometry          | Shapely, scikit-image                 |
| Evaluation        | IoU, Dice, Normalized Corner Distance |
| Deployment        | Single script inference               |

---

## 🎓 What This Demonstrates for the AI Trainee Role

✔ Ability to merge & standardize real heterogeneous datasets  
✔ Annotation & mask handling (COCO, polygons, JSON quadrilaterals)  
✔ Practical deep learning training pipeline engineering  
✔ Explainable results without black-box dependency  
✔ Optimization for limited VRAM GPUs (GTX 1650)  
✔ Clean modular structure & reproducible workflow  

---

## 📬 Contact

**Kirtiraj Chaudhari**  
Email: *available on request*  
GitHub: [https://github.com/KirtirajChaudhari](https://github.com/KirtirajChaudhari)

---

## ⭐ Final Summary

This repository demonstrates two scalable approaches to practical document segmentation without YOLO/SAM dependencies. The primary U-Net approach additionally provides corner estimation & rectification, while the DeepLabV3+ approach provides competitive segmentation quality.

---

### License

See [LICENSE](LICENSE) for details.