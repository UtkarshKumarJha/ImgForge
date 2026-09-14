"""Single-image inference utilities (used by the API and the old demo flow)."""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from data.ela import compute_ela
from model_factory import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


def load_model(checkpoint_path: str, device: str) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    arch = cfg.get("arch", "efficientnet_b0")
    input_mode = cfg.get("input_mode", "rgb_ela")

    model = build_model(arch, input_mode)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded — {arch}/{input_mode}, best val F1: {ckpt.get('val_f1', '?'):.4f} "
          f"(epoch {ckpt.get('epoch', '?')})")
    return model


def preprocess(image_path: str, device: str, image_size: int = IMAGE_SIZE) -> tuple:
    """Preprocess a single image into a 4-channel (RGB+ELA) tensor."""
    rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0

    rgb_resized = np.array(
        Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (image_size, image_size), Image.BILINEAR
        ), dtype=np.float32
    ) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    rgb_norm = (rgb_resized - mean) / std

    ela = compute_ela(image_path)
    ela_resized = np.array(
        Image.fromarray((ela * 255).astype(np.uint8)).resize(
            (image_size, image_size), Image.BILINEAR
        ), dtype=np.float32
    ) / 255.0
    ela_gray = ela_resized.mean(axis=2)
    ela_gray = (ela_gray - 0.5) / 0.5

    rgb_t = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
    ela_t = torch.from_numpy(ela_gray).unsqueeze(0).float()
    tensor = torch.cat([rgb_t, ela_t], dim=0).unsqueeze(0).to(device)

    return tensor, rgb_resized
