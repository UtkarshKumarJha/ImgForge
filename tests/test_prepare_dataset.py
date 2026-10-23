import os
from pathlib import Path
from data.prepare_dataset import collect_images, split_fingerprint


def test_collect_images_is_sorted_regardless_of_creation_order(tmp_path):
    d = tmp_path / "Au"
    d.mkdir()
    for name in ["c.jpg", "a.jpg", "b.jpg"]:
        (d / name).write_bytes(b"x")
    got = [Path(p).name for p, _ in collect_images(d, 0)]
    assert got == ["a.jpg", "b.jpg", "c.jpg"]


def test_split_fingerprint_reports_counts_and_hash(tmp_path):
    for split, labels in {"train": [0, 0, 1], "val": [0], "test": [1]}.items():
        with open(tmp_path / f"{split}.csv", "w") as f:
            f.write("path,label\n")
            for i, lab in enumerate(labels):
                f.write(f"/x/{split}_{i}.jpg,{lab}\n")
    fp = split_fingerprint(str(tmp_path))
    assert fp["counts"] == {"train": 3, "val": 1, "test": 1}
    assert fp["class_counts"]["train"] == {"0": 2, "1": 1}
    assert len(fp["sha256"]) == 64
