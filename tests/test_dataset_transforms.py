import numpy as np
import albumentations as A
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


def test_train_at_256_resizes_to_288_then_crops_to_256():
    spatial, _ = get_transforms("train", 256)
    resizes = [t for t in spatial.transforms if isinstance(t, A.Resize)]
    crops = [t for t in spatial.transforms if isinstance(t, A.RandomCrop)]
    assert [(t.height, t.width) for t in resizes] == [(288, 288)]
    assert [(t.height, t.width) for t in crops] == [(256, 256)]


def test_grid_dropout_never_touches_the_ela_stream():
    # It zeroes patches, i.e. exactly the regions ELA classifies (spec §2).
    for mode in ("ela", "rgb_ela"):
        spatial, pixel = get_transforms("train", 256, mode)
        assert not any(isinstance(t, A.GridDropout) for t in spatial.transforms), mode
        assert pixel is None, mode
    _, rgb_pixel = get_transforms("train", 256, "rgb")
    assert any(isinstance(t, A.GridDropout) for t in rgb_pixel.transforms)


def test_training_flips_horizontally_because_tta_averages_over_flips():
    spatial, _ = get_transforms("train", 256, "rgb_ela")
    assert any(isinstance(t, A.HorizontalFlip) for t in spatial.transforms)
    for split in ("val", "test"):
        assert not any(
            isinstance(t, A.HorizontalFlip) for t in get_transforms(split, 256)[0].transforms
        ), split


def test_random_crop_only_on_the_256_training_path():
    for spatial in (
        get_transforms("train", 224)[0],
        get_transforms("val", 256)[0],
        get_transforms("test", 256)[0],
    ):
        assert not any(isinstance(t, A.RandomCrop) for t in spatial.transforms)
