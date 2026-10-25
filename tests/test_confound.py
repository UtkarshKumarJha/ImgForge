import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from analysis.confound import mann_whitney, stratified_f1


def test_mann_whitney_separates_disjoint_quality_groups():
    r = mann_whitney([80, 82, 85, 88], [50, 55, 60, 65])
    assert r["p_value"] < 0.05
    assert r["median_a"] > r["median_b"]


def test_none_quality_values_are_excluded_not_coerced():
    r = mann_whitney([80, None, 85], [50, None, 60])
    assert r["n_a"] == 2 and r["n_b"] == 2


def test_stratified_f1_skips_single_class_and_small_bins():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0, 0, 1, 1, 1, 0])
    qf = [50, 50, 50, 80, 80, 80]      # two bins, each single-class
    r = stratified_f1(y, p, qf, min_per_class=1)
    assert r["usable_bins"] == 0
    assert r["insufficient_bins"] == 2


def test_stratified_f1_computes_when_a_bin_has_both_classes():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    p = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    qf = [50, 50, 50, 50, 60, 60, 60, 60]
    r = stratified_f1(y, p, qf, min_per_class=2)
    assert r["usable_bins"] == 2
    assert all(0.0 <= b["f1"] <= 1.0 for b in r["bins"])


def test_skipped_bins_do_not_enter_the_bin_macro_average():
    y = np.array([0, 0, 1, 1, 0, 0])
    p = np.array([0, 1, 1, 1, 0, 1])
    qf = [40, 40, 40, 40, 90, 90]      # 40 has both classes, 90 is all authentic
    r = stratified_f1(y, p, qf, min_per_class=2)
    assert r["usable_bins"] == 1
    assert r["bins"][1]["f1"] is None
    assert r["macro_f1"] == r["bins"][0]["f1"]


def test_jpeg_subset_f1_drops_non_jpeg_images():
    from analysis.confound import jpeg_subset_f1

    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0, 0, 1, 1, 1, 0])          # the last two are wrong
    kinds = ["jpeg", "jpeg", "jpeg", "jpeg", "not_jpeg", "unopenable"]
    r = jpeg_subset_f1(y, p, kinds, min_per_class=1)

    assert r["n"] == 4 and r["n_authentic"] == 2 and r["n_tampered"] == 2
    assert r["f1"] == 1.0                       # TIFF and unopenable images are excluded


def test_cli_writes_markdown_and_counts_unreadable_quality(tmp_path):
    imgs = []
    for i, (quality, label) in enumerate([(90, 0), (90, 0), (40, 1), (40, 1)]):
        f = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (16, 16), (i * 40, 100, 150)).save(f, quality=quality)
        imgs.append((str(f), label))
    bmp = tmp_path / "flat.bmp"
    Image.new("RGB", (16, 16)).save(bmp)      # no quantization table -> None
    imgs.append((str(bmp), 0))
    imgs.append((str(tmp_path / "gone.jpg"), 1))   # never written -> unopenable

    npz = tmp_path / "runA_test.npz"
    np.savez(npz, paths=np.array([p for p, _ in imgs]),
             labels=np.array([l for _, l in imgs]),
             probs=np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.4]),
             threshold=np.float64(0.42), tta=np.bool_(True))
    out = tmp_path / "confound.md"

    subprocess.run(
        [sys.executable, "-m", "analysis.confound", "--data-dir", "data",
         "--predictions", str(npz), "--out", str(out)],
        cwd=Path(__file__).resolve().parents[1], check=True,
        capture_output=True, text=True,
    )

    md = out.read_text()
    assert "runA_test" in md
    assert "Images analysed: 6" in md
    assert "No readable JPEG quality factor: 2" in md
    # A missing file must not be reported as "just a PNG".
    assert "not JPEG: 1, could not be opened: 1" in md
    assert "Decision threshold: 0.42" in md   # the artefact's own cut, not 0.5
    assert "across 0 usable bins" in md
    assert "insufficient" in md


def test_all_unopenable_images_warn_about_the_data_dir(tmp_path, capsys):
    from analysis.confound import analyse_run

    npz = tmp_path / "runB_test.npz"
    np.savez(npz, paths=np.array(["a/x.jpg", "b/x.jpg"]),
             labels=np.array([0, 1]), probs=np.array([0.1, 0.9]),
             threshold=np.float64(0.5), tta=np.bool_(False))
    r = analyse_run(str(npz), str(tmp_path / "wrong-dir"))

    assert r["n_unopenable"] == 2 and r["n_not_jpeg"] == 0
    assert "--data-dir" in capsys.readouterr().out


def test_absolute_quality_paths_are_not_rewritten_under_data_dir():
    from analysis.confound import _resolve

    assert _resolve("/nowhere/x.jpg", "/data") == "/nowhere/x.jpg"
