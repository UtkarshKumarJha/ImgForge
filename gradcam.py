#gradcam.py
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        Args:
            input_tensor: (1, 4, 224, 224) — single image with ELA channel
            class_idx: 1 = forged class
        Returns:
            heatmap: (H, W) float32 array in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global average pool gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam).squeeze().cpu().numpy()  # (H, W)

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.astype(np.float32)


def overlay_heatmap(original_image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap on original image.
    Args:
        original_image: (H, W, 3) uint8 RGB image
        cam: (H, W) float32 heatmap in [0, 1]
        alpha: blend factor
    Returns:
        overlay: (H, W, 3) uint8 RGB image
    """
    import matplotlib  # only the colour overlay needs matplotlib

    h, w = original_image.shape[:2]

    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (w, h))

    # Apply jet colormap
    colormap = matplotlib.colormaps["jet"]   # cm.get_cmap was removed in matplotlib 3.9
    heatmap_rgb = (colormap(cam_resized)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    overlay = (alpha * heatmap_rgb + (1 - alpha) * original_image).astype(np.uint8)
    return overlay


def parse_tamper_zone(cam: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Divides the CAM into 6 zones and returns the highest activation zone.
    Zones: top_left, top_right, mid_left, mid_right, bottom_left, bottom_right
    """
    h, w = cam.shape
    h3, w2 = h // 3, w // 2

    zones = {
        "top_left":     cam[:h3, :w2],
        "top_right":    cam[:h3, w2:],
        "mid_left":     cam[h3:2*h3, :w2],
        "mid_right":    cam[h3:2*h3, w2:],
        "bottom_left":  cam[2*h3:, :w2],
        "bottom_right": cam[2*h3:, w2:],
    }

    zone_scores = {name: float(region.mean()) for name, region in zones.items()}
    top_zone = max(zone_scores, key=zone_scores.get)

    return {
        "top_zone": top_zone,
        "zone_scores": zone_scores,
        "zone_confidence": round(zone_scores[top_zone], 4)
    }


def _infer_arch(model: nn.Module) -> str:
    """Recover the arch for legacy one-argument callers (app.py, api/main.py)."""
    from torchvision import models as tvm
    if isinstance(model, tvm.EfficientNet):
        return "efficientnet_b0"
    if isinstance(model, tvm.SwinTransformer):
        return "swin_tiny"
    return "resnet18_bilstm" if hasattr(model, "backbone") else "mlp"


def get_gradcam_target_layer(model: nn.Module, arch: str | None = None) -> nn.Module:
    """Return the last block that keeps a (B, C, H, W) conv feature map.

    Grad-CAM reduces gradients over dims (2, 3) as if the output were (B, C, H, W).
    Swin's blocks are channels-last (B, H, W, C) and the MLP has no conv trunk, so
    neither can produce a meaningful heatmap; both raise instead of returning a layer
    that would silently yield a meaningless map.
    """
    arch = arch or _infer_arch(model)
    if arch == "efficientnet_b0":
        return model.features[-1]
    if arch == "resnet18_bilstm":
        return model.backbone[-1][-1]
    if arch == "swin_tiny":
        raise ValueError(
            "Grad-CAM is not supported for arch 'swin_tiny': its blocks are channels-last "
            "(B, H, W, C), so a (B, C, H, W) Grad-CAM would be meaningless")
    if arch == "mlp":
        raise ValueError(
            "Grad-CAM is not supported for arch 'mlp': the model has no convolutional "
            "trunk to visualise")
    raise ValueError(f"Unknown arch {arch!r}")