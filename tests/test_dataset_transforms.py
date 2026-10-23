import numpy as np
from data.dataset import get_transforms


def test_rgb_keeps_photometric_augmentation():
    _, pixel = get_transforms("train", 256, "rgb")
    assert pixel is not None


def test_ela_modes_have_no_photometric_augmentation():
    for mode in ("ela", "rgb_ela"):
        _, pixel = get_transforms("train", 256, mode)
        assert pixel is None, mode


def test_spatial_transform_accepts_small_images_and_returns_square():
    spatial, _ = get_transforms("train", 256, "rgb_ela")
    rgb = np.random.randint(0, 255, (160, 240, 3), dtype=np.uint8)
    ela = np.random.randint(0, 255, (160, 240, 3), dtype=np.uint8)
    out = spatial(image=rgb, ela=ela)
    assert out["image"].shape[:2] == (256, 256)
    assert out["ela"].shape[:2] == (256, 256)
