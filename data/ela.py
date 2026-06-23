import cv2
import numpy as np
from PIL import Image
import io

def compute_ela(image_path: str, quality: int = 75, amplify: int = 10) -> np.ndarray:
    """
    Compute Error Level Analysis map for an image.
    
    Steps:
      1. Re-save the image at lower JPEG quality
      2. Reload the re-saved version
      3. Compute absolute difference between original and re-saved
      4. Amplify the difference for visibility
    
    Returns: ELA map as float32 array, shape (H, W, 3), values in [0, 1]
    """
    # Load original
    original = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)

    # Re-save at lower quality into memory buffer
    img_pil = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = np.array(Image.open(buffer).convert("RGB"), dtype=np.float32)

    # Absolute difference, amplified
    ela = np.abs(original - recompressed) * amplify
    ela = np.clip(ela, 0, 255) / 255.0  # normalize to [0, 1]

    return ela.astype(np.float32)


def get_4channel_tensor(image_path: str) -> np.ndarray:
    """
    Returns a (4, H, W) numpy array: [R, G, B, ELA_gray]
    ELA is converted to grayscale and used as the 4th channel.
    """
    rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    ela = compute_ela(image_path)                        # (H, W, 3)
    ela_gray = ela.mean(axis=2, keepdims=True)           # (H, W, 1)

    combined = np.concatenate([rgb, ela_gray], axis=2)   # (H, W, 4)
    return combined.transpose(2, 0, 1)                   # (4, H, W)