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

class _MLP(nn.Module):
    """16×16 adaptive-avg-pool → flatten → FC layers. Trained from scratch."""

    def __init__(self, in_channels: int, num_classes: int = 2,
                 pool_size: int = 16, hidden: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        flat_dim = in_channels * pool_size * pool_size
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def _build_mlp(in_channels: int, num_classes: int = 2) -> nn.Module:
    return _MLP(in_channels, num_classes)


# ── ResNet-18 + BiLSTM ───────────────────────────────────────────────────────

class _ResNet18BiLSTM(nn.Module):
    """ResNet-18 trunk → raster-flatten feature map → BiLSTM → classifier."""

    def __init__(self, backbone: nn.Module, lstm_hidden: int = 256,
                 num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        # ResNet-18 feature map: (B, 512, 7, 7) → 49 positions × 512 features
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=lstm_hidden,
            batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden * 2, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)                     # (B, 512, H, W)
        b, c, h, w = feat.shape
        seq = feat.reshape(b, c, h * w).permute(0, 2, 1)  # (B, H*W, 512)
        lstm_out, _ = self.lstm(seq)                # (B, H*W, 2*hidden)
        pooled = lstm_out.mean(dim=1)               # (B, 2*hidden)
        return self.classifier(pooled)


def _build_resnet18_bilstm(in_channels: int, num_classes: int = 2) -> nn.Module:
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.conv1 = _adapt_conv(resnet.conv1, in_channels)
    # Strip avgpool + fc to get the conv backbone only
    backbone = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
    )
    return _ResNet18BiLSTM(backbone, num_classes=num_classes)


# ── Swin-Tiny ────────────────────────────────────────────────────────────────

def _build_swin_tiny(in_channels: int, num_classes: int = 2) -> nn.Module:
    model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    # Patch embedding is features[0][0]: Conv2d(3, 96, 4×4, stride=4)
    model.features[0][0] = _adapt_conv(model.features[0][0], in_channels)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


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
