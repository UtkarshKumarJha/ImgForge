import torch
import torch.nn as nn
from torchvision import models
import os

CHECKPOINT = "models/best_model.pth"
DEVICE = "cpu"

def build_model():
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
        nn.Linear(in_features, 2)
    )
    return model


def export_to_onnx():
    print("Loading checkpoint...")
    model = build_model()
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model — val F1: {ckpt['val_f1']:.4f}")

    os.makedirs("models", exist_ok=True)
    onnx_path = "models/imgforge.onnx"

    dummy_input = torch.randn(1, 4, 224, 224)

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=17,
        do_constant_folding=True
    )

    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✓ ONNX export complete: {onnx_path} ({onnx_size:.2f} MB)")

    # Verify the exported model loads correctly
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verified — structure is valid")
    except ImportError:
        print("(Install 'onnx' package to run structure verification)")

    # Quick inference speed test
    try:
        import onnxruntime as ort
        import numpy as np
        import time

        session = ort.InferenceSession(onnx_path)
        test_input = np.random.randn(1, 4, 224, 224).astype(np.float32)

        # Warmup
        for _ in range(3):
            session.run(None, {"input": test_input})

        # Benchmark
        times = []
        for _ in range(20):
            start = time.time()
            session.run(None, {"input": test_input})
            times.append((time.time() - start) * 1000)

        avg_ms = sum(times) / len(times)
        print(f"✓ ONNX Runtime CPU inference: {avg_ms:.1f}ms average (20 runs)")

    except ImportError:
        print("(Install 'onnxruntime' package to run speed benchmark)")

    return onnx_path


if __name__ == "__main__":
    export_to_onnx()