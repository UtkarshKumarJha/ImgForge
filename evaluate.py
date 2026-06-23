import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os

from data.ela import compute_ela
from gradcam import GradCAM, overlay_heatmap, parse_tamper_zone, get_gradcam_target_layer

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224
CHECKPOINT    = "models/best_model.pth"


# ── Build model (must match train.py exactly) ─────────────────────────────────
def build_model(num_classes: int = 2) -> nn.Module:
    model = models.efficientnet_b0(weights=None)

    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    model.features[0][0] = new_conv

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


# ── Load checkpoint ───────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: str) -> nn.Module:
    model = build_model()
    ckpt = torch.load(checkpoint_path, map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded — best val F1: {ckpt['val_f1']:.4f} (epoch {ckpt['epoch']})")
    return model


# ── Preprocess a single image into 4-channel tensor ──────────────────────────
def preprocess(image_path: str, device: str) -> tuple:
    # Load RGB
    rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0

    # Resize
    rgb_resized = np.array(
        Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
        ), dtype=np.float32
    ) / 255.0

    # Normalize (ImageNet)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    rgb_norm = (rgb_resized - mean) / std  # (H, W, 3)

    # ELA channel
    ela = compute_ela(image_path)
    ela_resized = np.array(
        Image.fromarray((ela * 255).astype(np.uint8)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
        ), dtype=np.float32
    ) / 255.0
    ela_gray = ela_resized.mean(axis=2)
    ela_gray = (ela_gray - 0.5) / 0.5  # (H, W)

    # Stack to (1, 4, H, W)
    rgb_t = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
    ela_t = torch.from_numpy(ela_gray).unsqueeze(0).float()
    tensor = torch.cat([rgb_t, ela_t], dim=0).unsqueeze(0).to(device)

    return tensor, rgb_resized  # tensor for model, rgb for visualization


# ── Run inference + Grad-CAM on a single image ───────────────────────────────
def analyze(image_path: str, save_dir: str = "outputs"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = load_model(CHECKPOINT, device)

    # Set up Grad-CAM
    target_layer = get_gradcam_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    # Preprocess
    tensor, rgb_vis = preprocess(image_path, device)
    tensor.requires_grad_(True)

    # Forward pass
    with torch.enable_grad():
        output = model(tensor)

    probs = torch.softmax(output, dim=1)[0]
    pred_class = output.argmax(dim=1).item()
    confidence = probs[pred_class].item()

    label = "FORGED" if pred_class == 1 else "AUTHENTIC"
    forged_prob = probs[1].item()

    print(f"\n{'='*50}")
    print(f"Image     : {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
    print(f"Forged prob: {forged_prob:.4f}")

    # Grad-CAM
    cam = gradcam.generate(tensor, class_idx=1)
    zone_info = parse_tamper_zone(cam)

    print(f"Tamper Zone: {zone_info['top_zone']}")
    print(f"Zone Confidence: {zone_info['zone_confidence']}")
    print(f"Zone Scores: {zone_info['zone_scores']}")
    print(f"{'='*50}\n")

    # ── Visualization ─────────────────────────────────────────────────────────
    original_bgr = (rgb_vis * 255).astype(np.uint8)

    # ELA map
    ela_raw = compute_ela(image_path)
    ela_display = (ela_raw * 255).astype(np.uint8)
    ela_resized_display = cv2.resize(
        ela_display, (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Grad-CAM overlay
    heatmap_overlay = overlay_heatmap(original_bgr, cam, alpha=0.5)

    # Plot 3 panels: Original | ELA | Grad-CAM
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"DocForge Analysis — {label} ({confidence*100:.1f}% confidence)\n"
        f"Tamper Zone: {zone_info['top_zone']}",
        fontsize=13, fontweight="bold",
        color="red" if pred_class == 1 else "green"
    )

    axes[0].imshow(original_bgr)
    axes[0].set_title("Original Document", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(ela_resized_display)
    axes[1].set_title("ELA Map (Compression Artifacts)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(heatmap_overlay)
    axes[2].set_title(f"Grad-CAM Heatmap\n(Red = Suspicious Region)", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()

    # Save
    basename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f"{basename}_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to: {save_path}")

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "forged_prob": round(forged_prob, 4),
        "tamper_zone": zone_info["top_zone"],
        "zone_confidence": zone_info["zone_confidence"]
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <path_to_image>")
        print("Example: python evaluate.py data/raw/CASIA2.0_revised/Tp/Tp_D_CND_M_N_ani00018_ani00018_0279.jpg")
        sys.exit(1)

    result = analyze(sys.argv[1])