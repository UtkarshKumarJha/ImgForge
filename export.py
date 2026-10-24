import torch
import os

from model_factory import input_spec, load_model_from_checkpoint

CHECKPOINT = "models/best_model.pth"
DEVICE = "cpu"

def export_to_onnx():
    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = load_model_from_checkpoint(ckpt, DEVICE)
    model.eval()
    print(f"Loaded model — val F1: {ckpt['val_f1']:.4f}")

    os.makedirs("models", exist_ok=True)
    onnx_path = "models/imgforge.onnx"

    channels, image_size = input_spec(ckpt.get("cfg", {}))
    dummy_input = torch.randn(1, channels, image_size, image_size)

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
        test_input = np.random.randn(1, channels, image_size, image_size).astype(np.float32)

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