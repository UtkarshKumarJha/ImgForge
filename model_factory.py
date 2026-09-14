"""Model factory: build_model(arch, input_mode) → nn.Module."""

import torch
import torch.nn as nn
from torchvision import models

VALID_ARCHS = {"mlp", "efficientnet_b0", "resnet18_bilstm", "swin_tiny"}
VALID_MODES = {"rgb", "ela", "rgb_ela"}


def _input_channels(input_mode: str) -> int:
    return {"rgb": 3, "ela": 1, "rgb_ela": 4}[input_mode]


def _adapt_conv(old_conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Replace a pretrained Conv2d's input channels, copying/expanding weights."""
    if old_conv.in_channels == in_channels:
        return old_conv
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        if in_channels > old_conv.in_channels:
            new_conv.weight[:, :old_conv.in_channels] = old_conv.weight
            extra = in_channels - old_conv.in_channels
            new_conv.weight[:, old_conv.in_channels:] = (
                old_conv.weight.mean(dim=1, keepdim=True).expand(-1, extra, -1, -1)
            )
        else:
            new_conv.weight[:] = old_conv.weight[:, :in_channels]
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    return new_conv


# ── EfficientNet-B0 ─────────────────────────────────────────────────────────

def _build_efficientnet_b0(in_channels: int, num_classes: int = 2) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.features[0][0] = _adapt_conv(model.features[0][0], in_channels)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── MLP ──────────────────────────────────────────────────────────────────────

def _build_mlp(in_channels: int, num_classes: int = 2) -> nn.Module:
    raise NotImplementedError("mlp architecture pending (Phase 2)")

# ── ResNet-18 + BiLSTM ───────────────────────────────────────────────────────

def _build_resnet18_bilstm(in_channels: int, num_classes: int = 2) -> nn.Module:
    raise NotImplementedError("resnet18_bilstm architecture pending (Phase 2)")

# ── Swin-Tiny ────────────────────────────────────────────────────────────────

def _build_swin_tiny(in_channels: int, num_classes: int = 2) -> nn.Module:
    raise NotImplementedError("swin_tiny architecture pending (Phase 2)")

# ── Dispatch ─────────────────────────────────────────────────────────────────

_BUILDERS = {
    "efficientnet_b0": _build_efficientnet_b0,
    "mlp":             _build_mlp,
    "resnet18_bilstm": _build_resnet18_bilstm,
    "swin_tiny":       _build_swin_tiny,
}


def build_model(arch: str, input_mode: str, num_classes: int = 2) -> nn.Module:
    if arch not in VALID_ARCHS:
        raise ValueError(f"Unknown arch {arch!r}; choose from {VALID_ARCHS}")
    if input_mode not in VALID_MODES:
        raise ValueError(f"Unknown input_mode {input_mode!r}; choose from {VALID_MODES}")

    in_ch = _input_channels(input_mode)
    return _BUILDERS[arch](in_ch, num_classes)