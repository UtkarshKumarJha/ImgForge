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

def get_transforms(split: str, image_size: int = 224):
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.4),
            A.ImageCompression(quality_range=(50, 95), p=0.5),
            A.GridDropout(ratio=0.3, p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class ForgeryDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", image_size: int = 224):
        self.split = split
        self.transform = get_transforms(split, image_size)
        self.image_size = image_size
        self.samples = []

        csv_path = Path(data_dir) / f"{split}.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["path"], int(row["label"])))

        print(f"[{split}] Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        rgb = np.array(Image.open(img_path).convert("RGB"))
        augmented = self.transform(image=rgb)
        rgb_tensor = torch.from_numpy(
            augmented["image"].transpose(2, 0, 1)
        ).float()

        ela = compute_ela(img_path)
        ela_pil = Image.fromarray((ela * 255).astype(np.uint8))
        ela_resized = np.array(
            ela_pil.resize((self.image_size, self.image_size), Image.BILINEAR),
            dtype=np.float32
        ) / 255.0
        ela_gray = ela_resized.mean(axis=2)
        ela_gray = (ela_gray - 0.5) / 0.5
        ela_tensor = torch.from_numpy(ela_gray).unsqueeze(0).float()

        input_tensor = torch.cat([rgb_tensor, ela_tensor], dim=0)
        return input_tensor, torch.tensor(label, dtype=torch.long)