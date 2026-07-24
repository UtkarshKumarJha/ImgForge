# 🔍 ImgForge — Image Forgery Detection & Tamper Localization

**Deep Learning-powered image forensics for detecting manipulated images and identifying suspicious regions.**

ImgForge is an end-to-end image forgery detection system that classifies an image as **Authentic or Forged** and provides visual explanations of potentially manipulated regions using **Error Level Analysis (ELA)** and **Grad-CAM**.

The system uses a fine-tuned **EfficientNet-B0** with a custom **4-channel RGB + ELA input**, served through a **FastAPI inference API** and an interactive **React frontend**.

---

## 🎯 Problem It Solves

Digital images can be manipulated through techniques such as:

- Copy-move forgery
- Image splicing
- Object insertion/removal
- Inpainting
- Re-compression after editing

Detecting these modifications manually can be difficult because modern editing tools can produce visually convincing results.

ImgForge approaches this as a **binary image-forensics problem**:

> Given an image, determine whether it is authentic or manipulated and provide interpretable evidence showing which regions influenced the model's decision.

Instead of returning only a classification label, ImgForge combines deep learning with forensic visualization to make predictions easier to interpret.

---

## ✨ Features

- 🔍 **Image Forgery Detection** — classifies images as Authentic or Forged
- 📊 **Confidence Scoring** — displays classification confidence and class probabilities
- 🔬 **Error Level Analysis** — detects JPEG compression inconsistencies
- 🧠 **RGB + ELA Feature Fusion** — ELA is directly incorporated as a fourth model input channel
- 🗺️ **Grad-CAM Explainability** — visualizes regions influencing the forged prediction
- 📍 **Tamper Zone Localization** — divides Grad-CAM activations into six interpretable image regions
- ⚡ **GPU / CPU Inference** — automatically uses CUDA when available
- 🌐 **FastAPI Model Serving** — exposes the inference pipeline through a REST API
- ⚛️ **React Dashboard** — modern interface for image upload and forensic visualization

---

## 🧠 How ImgForge Works

```text
                     Uploaded Image
                           │
                           ▼
                    Image Validation
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
          RGB Image             Error Level Analysis
           3 Channels                1 Channel
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                  4-Channel Tensor
                   [R, G, B, ELA]
                           │
                           ▼
                   EfficientNet-B0
                           │
                    Classification
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
             AUTHENTIC             FORGED
                                      │
                                      ▼
                                   Grad-CAM
                                      │
                                      ▼
                           Suspicious Region Map
                                      │
                                      ▼
                         Six-Zone Localization
```

The React frontend sends the uploaded image to the FastAPI backend. The backend performs preprocessing, ELA generation, model inference, Grad-CAM generation, and tamper-zone analysis before returning the results to the frontend.

---

## 🔬 Model Architecture

### EfficientNet-B0 + ELA Fusion

ImgForge uses an **ImageNet-pretrained EfficientNet-B0** as its backbone.

A standard EfficientNet accepts:

```text
RGB → 3 channels
```

ImgForge modifies the first convolutional layer to accept:

```text
RGB + ELA → 4 channels
```

```text
Input
  │
  ├── Red
  ├── Green
  ├── Blue
  └── ELA
       │
       ▼
4 × 224 × 224 Tensor
       │
       ▼
Modified EfficientNet-B0
       │
       ▼
Binary Classification Head
       │
   ┌───┴────┐
   ▼        ▼
Authentic  Forged
```

This allows the network to learn simultaneously from **visual image content** and **compression-level forensic artifacts**.

---

## 🔬 Error Level Analysis

Error Level Analysis attempts to reveal differences in JPEG compression levels.

The image is:

1. Loaded and converted to RGB
2. Recompressed at a controlled JPEG quality
3. Compared with the original image
4. Pixel differences are amplified
5. The resulting ELA map is converted into the model's fourth input channel

Manipulated regions may exhibit compression characteristics different from surrounding image regions.

ELA is used as an additional forensic signal rather than as a standalone forgery detector.

---

## 🗺️ Grad-CAM Explainability

ImgForge uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize which spatial regions contribute most strongly toward the model's **Forged** class.

The activation map is divided into six regions:

```text
┌──────────────┬──────────────┐
│   Top Left   │  Top Right   │
├──────────────┼──────────────┤
│   Mid Left   │  Mid Right   │
├──────────────┼──────────────┤
│ Bottom Left  │ Bottom Right │
└──────────────┴──────────────┘
```

The region with the strongest mean activation is reported as the suspicious region for images classified as forged.

For images classified as authentic, the interface does not claim that a tampered region exists.

> Grad-CAM provides model explainability rather than pixel-perfect tamper segmentation.

---

## 📊 Model Results

| Metric | Result |
|---|---:|
| Best Validation Macro-F1 | **0.7767** |
| Validation Accuracy | **~77.6%** |
| Dataset | **CASIA 2.0** |
| Total Images | **12,614** |
| Authentic Images | **7,491** |
| Tampered Images | **5,123** |
| Input Resolution | **224 × 224** |
| Input Channels | **4 (RGB + ELA)** |

### Dataset Split

| Split | Total | Authentic | Tampered |
|---|---:|---:|---:|
| Train | 10,091 | 6,014 | 4,077 |
| Validation | 1,261 | 731 | 530 |
| Test | 1,262 | 746 | 516 |

The dataset contains multiple forms of image manipulation, including **splicing and copy-move forgery**.

---

## ⚙️ Training Strategy

The training pipeline includes:

- ImageNet-pretrained EfficientNet-B0
- 4-channel RGB + ELA input
- Data augmentation using Albumentations
- Class-weighted loss for dataset imbalance
- Macro-F1 based model selection
- Early stopping
- Weights & Biases experiment tracking
- Best-checkpoint persistence

Macro-F1 was prioritized over accuracy because the dataset contains an imbalance between authentic and tampered samples.

---

## 🛠️ Tech Stack

### Machine Learning

- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- NumPy
- OpenCV
- Pillow
- Albumentations
- Scikit-learn

### Explainability & Image Forensics

- Error Level Analysis (ELA)
- Grad-CAM
- Six-zone activation localization

### Backend

- FastAPI
- Uvicorn
- Python Multipart
- PyTorch inference

### Frontend

- React
- Vite
- JavaScript
- CSS
- Lucide React

### ML Engineering

- Weights & Biases
- ONNX
- ONNX Runtime
- CUDA acceleration

---

## 🌐 Application Architecture
<img width="1892" height="865" alt="image" src="https://github.com/user-attachments/assets/dfc166bf-01c4-4d14-929d-0c8ea6748edf" />
<img width="1872" height="691" alt="image" src="https://github.com/user-attachments/assets/a9c2a7a3-e2e4-4f4e-b27e-ab1cdd1911e1" />
<img width="1866" height="852" alt="image" src="https://github.com/user-attachments/assets/a3ba77bc-abda-45a6-a4d7-9c2e8e029932" />



```text
React Frontend
      │
      │ Image Upload
      │ multipart/form-data
      ▼
FastAPI REST API
      │
      ├── Image Validation
      ├── Preprocessing
      ├── ELA Generation
      │
      ▼
EfficientNet-B0
      │
      ├── Prediction
      └── Class Probabilities
      │
      ▼
Grad-CAM
      │
      └── Six-Zone Localization
      │
      ▼
FastAPI Response
      │
      ├── Verdict
      ├── Confidence
      ├── Probabilities
      ├── Tamper Zone
      ├── ELA Visualization
      └── Grad-CAM Visualization
      │
      ▼
React Forensic Dashboard
```

The model is loaded **once when the FastAPI server starts** and reused across inference requests to avoid repeatedly loading the checkpoint.

---

## 📂 Project Structure

```text
ImgForge/
│
├── api/
│   └── main.py                  # FastAPI inference API
│
├── data/
│   ├── ela.py                   # Error Level Analysis
│   ├── dataset.py               # PyTorch dataset
│   └── prepare_dataset.py       # CASIA dataset preparation
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React dashboard
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   └── package.json
│
├── models/
│   ├── best_model.pth           # PyTorch checkpoint
│   └── imgforge.onnx            # ONNX model
│
├── train.py                     # Training pipeline
├── evaluate.py                  # Model evaluation
├── gradcam.py                   # Grad-CAM + zone localization
├── export.py                    # ONNX export
├── validator.py
└── README.md
```

---

## 🚀 Running ImgForge Locally

### 1. Clone the repository

```bash
git clone https://github.com/MayurDas24/ImgForge.git
cd ImgForge
```

### 2. Install Python dependencies

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI backend

From the project root:

```bash
python -m uvicorn api.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 4. Start the React frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

Upload an image and select **Analyze Image** to run the complete forensic pipeline.

---

## 📡 API

### `POST /analyze`

Accepts an uploaded JPG/JPEG/PNG image and returns the forensic analysis.

Example response:

```json
{
  "filename": "sample.jpg",
  "verdict": "FORGED",
  "confidence": 90.63,
  "forged_probability": 90.63,
  "authentic_probability": 9.37,
  "tamper_zone": "Bottom Left",
  "zone_confidence": 18.92,
  "model": "EfficientNet-B0 + ELA",
  "device": "cuda"
}
```

The complete response additionally contains Base64-encoded **ELA** and **Grad-CAM** visualizations for rendering by the frontend.

---

## ⚡ Model Export

The trained PyTorch model can also be exported to ONNX:

```bash
python export.py
```

This enables CPU-oriented inference through ONNX Runtime and provides a path toward production inference optimization.

---

## ⚠️ Limitations

ImgForge is an experimental image-forensics system and should not be treated as a definitive authenticity verification tool.

Current limitations include:

- Training is primarily based on CASIA 2.0
- Performance may decrease on images from significantly different distributions
- ELA is primarily meaningful for compression-based forensic analysis
- Grad-CAM provides coarse model explainability rather than pixel-level segmentation
- The six-zone localization identifies high-activation regions rather than exact tampered boundaries
- Model confidence should not be interpreted as forensic certainty

---

## 🔮 Future Improvements

- **Pixel-level localization** using a segmentation architecture such as U-Net
- **Domain adaptation for identity documents** such as PAN and Aadhaar
- **Confidence calibration** using temperature scaling
- **TensorRT deployment** for lower-latency GPU inference
- Larger and more diverse forgery datasets
- Dedicated synthetic document-tampering pipeline
- Structural document validation using OCR and field-level consistency checks
- Improved localization evaluation against ground-truth tamper masks

---

## 💡 Why I Built ImgForge

ImgForge was built to explore the intersection of **deep learning, computer vision, image forensics, explainable AI, and ML deployment**.

Rather than stopping at model training, the project implements the complete workflow:

**dataset preparation → forensic feature engineering → model training → evaluation → explainability → model serving → React application**

The goal was to understand how a computer vision model can be transformed into an interpretable, end-to-end ML application.

---

## 👤 Author

**Mayur Das**  
B.Tech — Computer and Communication Engineering  
MIT Manipal, MAHE · 2023–2027

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ If you found ImgForge interesting, consider starring the repository.
