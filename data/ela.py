import hashlib
import io
import os

import numpy as np
from PIL import Image


def _ela_cache_path(image_path: str, quality: int) -> str | None:
    cache_dir = os.environ.get("IMGFORGE_ELA_CACHE")
    if not cache_dir:
        return None
    raw = f"{image_path}|{quality}"
    key = hashlib.sha256(raw.encode()).hexdigest()
    return os.path.join(cache_dir, f"{key}.npy")


def read_jpeg_quality(image_path: str) -> int | None:
    """Read the JPEG quantization-table quality estimate. Returns None for non-JPEG."""
    try:
        img = Image.open(image_path)
        qtables = img.quantization
        if not qtables:
            return None
        # Estimate quality from the luminance table (table 0)
        # PIL stores tables as lists of 64 ints; lower values = higher quality
        lum = list(qtables[0]) if 0 in qtables else list(next(iter(qtables.values())))
        avg = sum(lum) / len(lum)
        # Rough inverse mapping: avg≈1→q≈100, avg≈25→q≈50, avg≈50→q≈25
        if avg <= 1:
            return 100
        q = max(1, min(100, int(100 - (avg - 1) * 1.6)))
        return q
    except Exception:
        return None


def compute_ela(image_path: str, quality: int = 75, amplify: int = 10) -> np.ndarray:
    """Compute Error Level Analysis map for an image.

    Returns: ELA map as float32 array, shape (H, W, 3), values in [0, 1].
    Caches to disk as uint8 when IMGFORGE_ELA_CACHE env var is set.
    """
    cp = _ela_cache_path(image_path, quality)
    if cp and os.path.exists(cp):
        try:
            return np.load(cp).astype(np.float32) / 255.0
        except Exception:
            os.remove(cp)

    original = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)

    img_pil = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = np.array(Image.open(buffer).convert("RGB"), dtype=np.float32)

    ela = np.abs(original - recompressed) * amplify
    ela = np.clip(ela, 0, 255)

    if cp:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        np.save(cp, ela.astype(np.uint8))

    return (ela / 255.0).astype(np.float32)
