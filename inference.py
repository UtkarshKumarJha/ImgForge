"""Single-image inference utilities (used by the API and the old demo flow)."""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from data.dataset import get_transforms
from data.ela import compute_ela
from model_factory import _input_channels, load_model_from_checkpoint

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224
INPUT_MODE    = "rgb_ela"


def load_model(checkpoint_path: str, device: str) -> tuple[nn.Module, dict]:
    """Load the checkpoint's weights and return (model, cfg) for preprocessing."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    arch = cfg.get("arch", "efficientnet_b0")
    input_mode = cfg.get("input_mode", "rgb_ela")

    model = load_model_from_checkpoint(ckpt, device)
    model.eval()
    print(f"Model loaded — {arch}/{input_mode}, best val F1: {ckpt.get('val_f1', '?'):.4f} "
          f"(epoch {ckpt.get('epoch', '?')})")
    return model, cfg


def preprocess(image_path: str, device: str, image_size: int = IMAGE_SIZE,
               input_mode: str = INPUT_MODE) -> tuple:
    """Preprocess a single image into the checkpoint's channel layout."""
    # Same steps as ForgeryDataset at evaluation time, so the demo scores what the model saw.
    rgb = np.array(Image.open(image_path).convert("RGB"))
    ela = (compute_ela(image_path, use_cache=False) * 255).astype(np.uint8)
    out = get_transforms("val", image_size)[0](image=rgb, ela=ela)
    rgb_resized = out["image"].astype(np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    rgb_norm = (rgb_resized - mean) / std

    ela_gray = out["ela"].astype(np.float32).mean(axis=2) / 255.0
    ela_gray = (ela_gray - 0.5) / 0.5

    rgb_t = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
    ela_t = torch.from_numpy(ela_gray).unsqueeze(0).float()
    channels = _input_channels(input_mode)
    parts = ([rgb_t] if channels != 1 else []) + ([ela_t] if channels != 3 else [])
    tensor = torch.cat(parts, dim=0).unsqueeze(0).to(device)

    return tensor, rgb_resized
