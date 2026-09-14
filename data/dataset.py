import csv
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
from data.ela import compute_ela

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _input_channels(input_mode: str) -> int:
    return {"rgb": 3, "ela": 1, "rgb_ela": 4}[input_mode]


def get_transforms(split: str, image_size: int = 224):
    """Return (spatial_transform, pixel_transform) pair.

    spatial_transform is applied to both RGB and ELA (via additional_targets).
    pixel_transform is applied to RGB only (photometric augmentations that
    would destroy the ELA forensic signal).
    """
    if split == "train":
        spatial = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.4),
                A.GridDropout(ratio=0.3, p=0.2),
            ],
            additional_targets={"ela": "image"},
        )
        pixel = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.ImageCompression(quality_range=(50, 95), p=0.5),
        ])
    else:
        spatial = A.Compose(
            [A.Resize(image_size, image_size)],
            additional_targets={"ela": "image"},
        )
        pixel = None

    return spatial, pixel


class ForgeryDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train",
                 image_size: int = 224, input_mode: str = "rgb_ela",
                 ela_quality: int = 75):
        self.split = split
        self.spatial_transform, self.pixel_transform = get_transforms(split, image_size)
        self.image_size = image_size
        self.input_mode = input_mode
        self.ela_quality = ela_quality
        self.samples = []

        csv_path = Path(data_dir) / f"{split}.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["path"], int(row["label"])))

        print(f"[{split}] Loaded {len(self.samples)} samples (mode={input_mode})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        rgb = np.array(Image.open(img_path).convert("RGB"))

        need_ela = self.input_mode in ("ela", "rgb_ela")
        if need_ela:
            ela_raw = compute_ela(img_path, quality=self.ela_quality)
            ela_uint8 = (ela_raw * 255).astype(np.uint8)
        else:
            ela_uint8 = None

        augmented = self.spatial_transform(image=rgb, ela=ela_uint8) if need_ela \
            else self.spatial_transform(image=rgb, ela=rgb)
        rgb_aug = augmented["image"]

        if self.pixel_transform is not None:
            rgb_aug = self.pixel_transform(image=rgb_aug)["image"]

        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        rgb_norm = (rgb_aug.astype(np.float32) / 255.0 - mean) / std
        rgb_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()

        if need_ela:
            ela_aug = augmented["ela"]
            ela_f = ela_aug.astype(np.float32) / 255.0
            ela_gray = ela_f.mean(axis=2)
            ela_gray = (ela_gray - 0.5) / 0.5
            ela_tensor = torch.from_numpy(ela_gray).unsqueeze(0).float()

        if self.input_mode == "rgb_ela":
            return torch.cat([rgb_tensor, ela_tensor], dim=0), torch.tensor(label, dtype=torch.long)
        elif self.input_mode == "ela":
            return ela_tensor, torch.tensor(label, dtype=torch.long)
        else:
            return rgb_tensor, torch.tensor(label, dtype=torch.long)
