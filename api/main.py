import os
import sys
import base64
import tempfile
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evaluate import load_model, preprocess
from data.ela import compute_ela
from gradcam import (
    GradCAM,
    overlay_heatmap,
    parse_tamper_zone,
    get_gradcam_target_layer,
)


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="ImgForge API",
    description="Deep Learning Image Forgery Detection & Tamper Localization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = os.path.join(
    ROOT_DIR,
    "models",
    "best_model.pth"
)

print(f"Using device: {DEVICE}")
print("Loading ImgForge model...")

model = load_model(CHECKPOINT, DEVICE)

target_layer = get_gradcam_target_layer(model)
gradcam = GradCAM(model, target_layer)

print("ImgForge model ready.")


# ---------------------------------------------------------
# Helper: numpy image -> base64
# ---------------------------------------------------------

def numpy_to_base64(image: np.ndarray) -> str:
    """
    Convert RGB numpy image to a base64 PNG
    that can be displayed directly by React.
    """

    image = np.clip(image, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "ImgForge API",
        "model": "EfficientNet-B0 + ELA",
        "device": DEVICE,
    }


# ---------------------------------------------------------
# Analysis endpoint
# ---------------------------------------------------------

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/jpg",
        "image/png",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    temp_path = None

    try:

        # -------------------------------------------------
        # Read uploaded image
        # -------------------------------------------------

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        # -------------------------------------------------
        # Save temporarily
        # Existing pipeline expects image paths
        # -------------------------------------------------

        suffix = os.path.splitext(file.filename or "")[1]

        if not suffix:
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        # Validate image
        try:
            Image.open(temp_path).verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image."
            )

        # -------------------------------------------------
        # Preprocessing
        # RGB + ELA -> 4 channel tensor
        # -------------------------------------------------

        tensor, rgb_vis = preprocess(
            temp_path,
            DEVICE
        )

        tensor.requires_grad_(True)

        # -------------------------------------------------
        # EfficientNet inference
        # -------------------------------------------------

        with torch.enable_grad():

            output = model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

            pred_class = output.argmax(
                dim=1
            ).item()

        confidence = probabilities[pred_class].item()
        forged_probability = probabilities[1].item()
        authentic_probability = probabilities[0].item()

        label = (
            "FORGED"
            if pred_class == 1
            else "AUTHENTIC"
        )

        # -------------------------------------------------
        # Grad-CAM
        # Always explain forged class
        # -------------------------------------------------

        cam = gradcam.generate(
            tensor,
            class_idx=1
        )

        zone_info = parse_tamper_zone(cam)

        # -------------------------------------------------
        # ELA visualization
        # -------------------------------------------------

        ela_raw = compute_ela(temp_path)

        ela_display = (
            ela_raw * 255
        ).astype(np.uint8)

        # -------------------------------------------------
        # Grad-CAM visualization
        # -------------------------------------------------

        original_rgb = (
            rgb_vis * 255
        ).astype(np.uint8)

        heatmap_overlay = overlay_heatmap(
            original_rgb,
            cam,
            alpha=0.5
        )

        # -------------------------------------------------
        # Convert visualizations for React
        # -------------------------------------------------

        ela_base64 = numpy_to_base64(
            ela_display
        )

        gradcam_base64 = numpy_to_base64(
            heatmap_overlay
        )

        # -------------------------------------------------
        # Clean zone name
        # -------------------------------------------------

        tamper_zone = zone_info[
            "top_zone"
        ].replace("_", " ").title()

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return {

            "filename": file.filename,

            "verdict": label,

            "confidence": round(
                confidence * 100,
                2
            ),

            "forged_probability": round(
                forged_probability * 100,
                2
            ),

            "authentic_probability": round(
                authentic_probability * 100,
                2
            ),

            "tamper_zone": tamper_zone,

            "zone_confidence": round(
                zone_info["zone_confidence"] * 100,
                2
            ),

            "zone_scores": zone_info[
                "zone_scores"
            ],

            "ela_image": ela_base64,

            "gradcam_image": gradcam_base64,

            "model": "EfficientNet-B0 + ELA",

            "device": DEVICE,
        }

    except HTTPException:
        raise

    except Exception as e:

        print("Analysis error:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(e)}"
        )

    finally:

        # -------------------------------------------------
        # Remove temporary uploaded image
        # -------------------------------------------------

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)