import csv
import hashlib
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
from tqdm import tqdm
from data.ela import compute_ela

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Directory of pre-resized uint8 RGB/ELA arrays written by build_array_cache().
ARRAY_CACHE_ENV = "IMGFORGE_ARRAY_CACHE"


def _input_channels(input_mode: str) -> int:
    return {"rgb": 3, "ela": 1, "rgb_ela": 4}[input_mode]


def get_transforms(split: str, image_size: int = 224, input_mode: str = "rgb"):
    """Return (spatial_transform, pixel_transform) pair.

    spatial_transform is applied to both RGB and ELA (via additional_targets).
    pixel_transform is applied to RGB only (photometric augmentations that
    would destroy the ELA forensic signal).
    """
    if split == "train":
        resize = (
            [A.Resize(288, 288), A.RandomCrop(256, 256)]
            if image_size == 256
            else [A.Resize(image_size, image_size)]
        )
        spatial = A.Compose(
            resize + [
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.4),
            ],
            additional_targets={"ela": "image"},
        )
        # GridDropout punches holes in the very regions ELA measures, so it is
        # RGB-only (spec §2). It is geometric, so it cannot live in the pixel pass.
        pixel = A.Compose([
            A.GridDropout(ratio=0.3, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.ImageCompression(quality_range=(50, 95), p=0.5),
        ]) if input_mode == "rgb" else None
    else:
        spatial = A.Compose(
            [A.Resize(image_size, image_size)],
            additional_targets={"ela": "image"},
        )
        pixel = None

    return spatial, pixel


def _read_split(data_dir: str, split: str) -> list[tuple[str, int]]:
    with open(Path(data_dir) / f"{split}.csv", newline="") as f:
        return [(row["path"], int(row["label"])) for row in csv.DictReader(f)]


def _cache_stem(cache_dir: str, split: str, samples, image_size: int, ela_quality: int) -> str:
    """Cache file prefix; the digest of the path list keeps a different split from matching."""
    digest = hashlib.sha1("\n".join(p for p, _ in samples).encode()).hexdigest()[:10]
    return os.path.join(cache_dir, f"{split}_{digest}_{image_size}px_q{ela_quality}")


def _resized_pair(job):
    path, image_size, ela_quality = job
    rgb = np.array(Image.open(path).convert("RGB"))
    ela = (compute_ela(path, quality=ela_quality, use_cache=False) * 255).astype(np.uint8)
    out = get_transforms("val", image_size)[0](image=rgb, ela=ela)  # Resize(S, S) only
    return out["image"], out["ela"]


def build_array_cache(data_dir: str, image_size: int, ela_quality: int = 75,
                      cache_dir: str | None = None, workers: int | None = None):
    """Precompute the Resize(S, S) RGB and ELA of every split as uint8 .npy arrays.

    Every transform pipeline starts with that deterministic resize (except 256 px
    training), so caching its output leaves the batches unchanged while each sample
    stops decoding the full-size file and recomputing ELA every epoch.
    """
    cache_dir = cache_dir or os.environ[ARRAY_CACHE_ENV]
    os.makedirs(cache_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        samples = _read_split(data_dir, split)
        stem = _cache_stem(cache_dir, split, samples, image_size, ela_quality)
        if os.path.exists(stem + "_ela.npy"):
            continue
        shape = (len(samples), image_size, image_size, 3)
        arrays = {k: np.lib.format.open_memmap(f"{stem}_{k}.tmp.npy", mode="w+",
                                               dtype=np.uint8, shape=shape)
                  for k in ("rgb", "ela")}
        jobs = [(p, image_size, ela_quality) for p, _ in samples]
        with Pool(workers) as pool:
            pairs = pool.imap(_resized_pair, jobs, chunksize=32)
            for i, (rgb, ela) in enumerate(tqdm(pairs, total=len(jobs),
                                                desc=f"array cache {split} {image_size}px")):
                arrays["rgb"][i], arrays["ela"][i] = rgb, ela
        for k in ("rgb", "ela"):  # ela last: its file marks a complete cache
            arrays.pop(k).flush()
            os.replace(f"{stem}_{k}.tmp.npy", f"{stem}_{k}.npy")


class ForgeryDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train",
                 image_size: int = 224, input_mode: str = "rgb_ela",
                 ela_quality: int = 75):
        self.split = split
        self.spatial_transform, self.pixel_transform = get_transforms(split, image_size, input_mode)
        self.image_size = image_size
        self.input_mode = input_mode
        self.ela_quality = ela_quality
        self.samples = _read_split(data_dir, split)

        self._stem, self._rgb, self._ela = None, None, None
        cache_dir = os.environ.get(ARRAY_CACHE_ENV)
        # 256 px training resizes to 288 before cropping, so it cannot start from the cache.
        if cache_dir and not (split == "train" and image_size == 256):
            stem = _cache_stem(cache_dir, split, self.samples, image_size, ela_quality)
            if os.path.exists(stem + "_ela.npy"):
                self._stem = stem

        cached = ", array cache" if self._stem else ""
        print(f"[{split}] Loaded {len(self.samples)} samples (mode={input_mode}{cached})")

    def __getstate__(self):
        # Each DataLoader worker maps the files itself; pickling a map copies the array.
        return {**self.__dict__, "_rgb": None, "_ela": None}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        need_ela = self.input_mode in ("ela", "rgb_ela")

        if self._stem:
            if self._rgb is None:
                self._rgb = np.load(self._stem + "_rgb.npy", mmap_mode="r")
                self._ela = np.load(self._stem + "_ela.npy", mmap_mode="r")
            rgb = np.array(self._rgb[idx])
            ela_uint8 = np.array(self._ela[idx]) if need_ela else None
        else:
            rgb = np.array(Image.open(img_path).convert("RGB"))
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
