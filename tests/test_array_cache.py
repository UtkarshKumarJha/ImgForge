import numpy as np
import torch
from PIL import Image

from data.dataset import ARRAY_CACHE_ENV, ForgeryDataset, build_array_cache


def _split(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    for split, sizes in {"train": [(160, 240), (300, 200)], "val": [(256, 384)],
                         "test": [(120, 90)]}.items():
        rows = []
        for i, (h, w) in enumerate(sizes):
            p = tmp_path / f"{split}_{i}.{'jpg' if i % 2 == 0 else 'tif'}"
            Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(p)
            rows.append(f"{p},{i % 2}")
        (data / f"{split}.csv").write_text("path,label\n" + "\n".join(rows) + "\n")
    return data


def test_cached_samples_match_the_uncached_pipeline(tmp_path, monkeypatch):
    data = _split(tmp_path)
    monkeypatch.delenv(ARRAY_CACHE_ENV, raising=False)
    plain = [ForgeryDataset(str(data), s, 96, "rgb_ela")[0][0] for s in ("val", "test")]

    build_array_cache(str(data), 96, cache_dir=str(tmp_path / "cache"), workers=2)
    monkeypatch.setenv(ARRAY_CACHE_ENV, str(tmp_path / "cache"))
    for s, ref in zip(("val", "test"), plain):
        ds = ForgeryDataset(str(data), s, 96, "rgb_ela")
        assert ds._stem is not None, s
        assert torch.equal(ds[0][0], ref), s

    train = ForgeryDataset(str(data), "train", 96, "ela")
    assert train._stem is not None
    assert train[1][0].shape == (1, 96, 96)


def test_cache_for_another_size_or_split_is_not_used(tmp_path, monkeypatch):
    data = _split(tmp_path)
    build_array_cache(str(data), 96, cache_dir=str(tmp_path / "cache"), workers=1)
    monkeypatch.setenv(ARRAY_CACHE_ENV, str(tmp_path / "cache"))
    assert ForgeryDataset(str(data), "val", 128, "ela")._stem is None
    (data / "val.csv").write_text((data / "test.csv").read_text())
    assert ForgeryDataset(str(data), "val", 96, "ela")._stem is None
