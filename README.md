# 🔍 ImgForge

> **Image Forgery Detection & Tamper Localization using Deep Learning**

A forgery detection system that classifies images as **Authentic or Forged** and pinpoints *exactly where* the tampering occurred — using EfficientNet-B0 with Error Level Analysis fusion and Grad-CAM localization.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=flat-square&logo=streamlit)

---

## 🎯 What It Does

Upload any image → ImgForge returns:

- ✅ / 🚨 **Verdict** — Authentic or Forged
- 📊 **Confidence score** — model output probability
- 🗺️ **Grad-CAM heatmap** — visual overlay showing the suspicious region
- 📍 **Tamper zone** — which of 6 image zones was flagged
- 🔬 **ELA map** — raw compression artifact analysis

---

## 🏗️ Architecture
Input Image
│
├── RGB channels (3)
│
├── ELA Map ──→ Compression artifact channel (1)
│
└── 4-Channel Tensor [R, G, B, ELA]
│
▼
EfficientNet-B0
(pretrained ImageNet, 4-ch input adapter)
│
├── Binary Classification Head
│         └── Authentic / Forged
│
└── Grad-CAM Hooks
└── Tamper Localization Heatmap

---

## 🔬 Key Engineering Decisions

**1. ELA as a 4th Input Channel**
Error Level Analysis detects JPEG re-encoding artifacts — tampered regions compress differently from authentic ones. ImgForge fuses ELA directly as a 4th channel alongside RGB rather than treating it as a separate post-processing step. The first conv layer of EfficientNet-B0 is expanded from 3→4 channels via pretrained weight duplication, preserving ImageNet pretraining benefits.

**2. Grad-CAM Tamper Localization**
Hooks on the final convolutional block produce a spatial activation map showing which regions most influenced the forgery prediction. The map is divided into 6 zones for interpretable, structured reporting rather than a raw heatmap alone.

**3. Class-Weighted Loss**
CASIA 2.0 has a ~60:40 authentic:tampered ratio. Class weights are computed dynamically from the training split to prevent majority-class bias.

**4. Early Stopping on Macro F1**
Training monitors macro F1 (not accuracy) to avoid bias toward the majority class — critical for imbalanced forgery detection where false negatives are costly.

---

## 📊 Results

| Metric | Value |
|---|---|
| Best Validation F1 | 0.7767 |
| Validation Accuracy | 77.6% |
| Training Dataset | CASIA 2.0 (12,614 images) |
| Inference (PyTorch, GPU) | ~50ms |
| Inference (ONNX, CPU) | ~150-200ms |

---

## 🗂️ Dataset

**CASIA 2.0** — 12,614 images (7,491 authentic + 5,123 tampered)

| Split | Total | Authentic | Tampered |
|---|---|---|---|
| Train | 10,091 | 6,014 | 4,077 |
| Val | 1,261 | 731 | 530 |
| Test | 1,262 | 746 | 516 |

Forgery types covered: copy-move, splicing, inpainting.

Download: [CASIA 2.0 on Kaggle](https://www.kaggle.com/datasets/divg07/casia-20-image-forgery-detection-dataset)

---

## 🚀 Quick Start

```bash
git clone https://github.com/MayurDas24/ImgForge.git
cd ImgForge

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install albumentations opencv-python pillow numpy scikit-learn matplotlib tqdm wandb streamlit onnx onnxruntime

# Download CASIA 2.0 to data/raw/CASIA2.0_revised/
python data/prepare_dataset.py

# Train
python train.py

# Evaluate on a single image
python evaluate.py path/to/image.jpg

# Export to ONNX
python export.py

# Launch demo UI
streamlit run app.py
```

---

## 📁 Repository Structure
ImgForge/
├── data/
│   ├── init.py
│   ├── ela.py               # ELA computation
│   ├── dataset.py           # PyTorch Dataset with 4-channel input
│   └── prepare_dataset.py   # CSV index generator
├── models/                  # Saved checkpoints (not tracked)
├── app.py                   # Streamlit demo UI
├── train.py                 # Training loop with W&B logging
├── evaluate.py              # Single image inference + visualization
├── gradcam.py                # Grad-CAM implementation + zone parser
├── export.py                 # ONNX export + benchmarking
└── README.md

---

## 🔮 Future Work

- **Domain adaptation to ID documents** — fine-tune on labeled Indian government document forgeries (Aadhaar, PAN) once a sufficiently large labeled dataset is available; explored a synthetic tampering pipeline using OpenCV/PIL as a proof of concept
- **Pixel-level segmentation** — U-Net decoder for pixel-accurate tamper maps instead of zone-level Grad-CAM
- **Confidence calibration** — temperature scaling to ensure reported confidence reflects true accuracy
- **TensorRT deployment** — GPU inference under 20ms for video-frame analysis
- **Structural document validation layer** — rule-based OCR field/format checks (PAN regex, Aadhaar Verhoeff checksum) as a complementary signal alongside pixel forensics

---

## 🧠 Technical Stack

| Component | Technology |
|---|---|
| Training Framework | PyTorch 2.5 |
| Model Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Forensic Feature | ELA — Error Level Analysis |
| Explainability | Grad-CAM (custom implementation) |
| Augmentation | Albumentations |
| Experiment Tracking | Weights & Biases |
| Model Export | ONNX |
| Demo UI | Streamlit |

---

## 👤 Author

**Mayur Das**
B.Tech CCE · MIT Manipal, MAHE · Batch 2023–2027

[GitHub](https://github.com/MayurDas24) · [LinkedIn](https://linkedin.com/in/mayurrdas24) · [LeetCode](https://leetcode.com/MayurDas_)

---

*Deep learning portfolio project exploring image forensics and forgery detection.*
