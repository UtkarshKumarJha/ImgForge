import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from model_factory import build_model
from gradcam import get_gradcam_target_layer
from analysis.localization import cam_to_mask, iou_dice, choose_threshold, coverage


def test_perfect_and_disjoint_masks():
    gt = np.zeros((8, 8), bool); gt[2:6, 2:6] = True
    assert iou_dice(gt, gt) == (1.0, 1.0)
    assert iou_dice(~gt, gt) == (0.0, 0.0)


def test_empty_ground_truth_returns_zero_not_nan():
    gt = np.zeros((8, 8), bool)
    iou, dice = iou_dice(np.zeros((8, 8), bool), gt)
    assert iou == 0.0 and dice == 0.0


def test_threshold_selection_picks_the_separable_value():
    cam = np.zeros((8, 8)); cam[2:6, 2:6] = 1.0
    gt = np.zeros((8, 8), bool); gt[2:6, 2:6] = True
    best, score = choose_threshold([cam], [gt], np.array([0.3, 0.5, 0.7]))
    assert best in (0.3, 0.5, 0.7) and score == 1.0


def test_coverage_reports_fraction_not_exception():
    assert coverage(["a", "b", "c"], ["a"]) == 1 / 3


def test_cam_to_mask_keeps_the_cams_own_resolution():
    cam = np.linspace(0, 1, 16).reshape(4, 4)
    mask = cam_to_mask(cam, 0.5)
    assert mask.shape == (4, 4) and mask.dtype == bool
    assert not mask[0, 0] and mask[-1, -1]
    assert cam_to_mask(np.array([0.5]), 0.5).all()   # threshold is inclusive


def test_threshold_ties_resolve_to_the_smallest_grid_value():
    cam = np.zeros((4, 4)); cam[:2, :2] = 1.0
    gt = np.zeros((4, 4), bool); gt[:2, :2] = True
    best, score = choose_threshold([cam], [gt], np.array([0.9, 0.1, 0.5]))
    assert best == 0.1 and score == 1.0


def test_threshold_selection_ignores_empty_ground_truth_pairs():
    cam = np.zeros((4, 4)); cam[:2, :2] = 1.0
    gt = np.zeros((4, 4), bool); gt[:2, :2] = True
    empty = np.zeros((4, 4), bool)
    best, score = choose_threshold([cam, cam], [empty, gt], np.array([0.5]))
    assert best == 0.5 and score == 1.0


def test_threshold_selection_without_ground_truth_reports_none():
    cam = np.zeros((4, 4))
    empty = np.zeros((4, 4), bool)
    assert choose_threshold([cam], [empty], np.array([0.3, 0.5])) == (None, 0.0)


def test_coverage_is_empty_safe_and_matches_stems_exactly():
    assert coverage([], ["a.png"]) == 0.0
    assert coverage(["X/IMG_1.JPG"], ["m/img_1.png"]) == 1.0
    # No suffix stripping: a mask named with an extra suffix is not a match.
    assert coverage(["X/IMG_1.JPG"], ["m/IMG_1_mask.png"]) == 0.0


def test_gradcam_target_layer_accepts_explicit_and_inferred_arch():
    eff = build_model("efficientnet_b0", "rgb_ela")
    assert get_gradcam_target_layer(eff, "efficientnet_b0") is eff.features[-1]
    assert get_gradcam_target_layer(eff) is eff.features[-1]

    res = build_model("resnet18_bilstm", "rgb_ela")
    assert get_gradcam_target_layer(res, "resnet18_bilstm") is res.backbone[-1][-1]


@pytest.mark.parametrize("arch", ["swin_tiny", "mlp"])
def test_gradcam_target_layer_refuses_arches_without_a_spatial_conv_trunk(arch):
    model = build_model(arch, "rgb_ela")
    with pytest.raises(ValueError) as exc:
        get_gradcam_target_layer(model, arch)
    assert arch in str(exc.value)


def test_cli_reports_zero_coverage_and_exits_zero(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for split in ("test", "val"):
        (data / f"{split}.csv").write_text(
            "path,label\nimg/a.jpg,0\nimg/t1.jpg,1\nimg/t2.jpg,1\n")
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "unrelated_mask.png").write_bytes(b"")
    out = tmp_path / "localization.md"

    r = subprocess.run(
        [sys.executable, "-m", "analysis.localization",
         "--ckpt", str(tmp_path / "never_loaded.pth"),
         "--data-dir", str(data), "--masks-dir", str(masks),
         "--split", "test", "--out", str(out)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )

    assert r.returncode == 0, r.stderr
    md = out.read_text()
    assert "could not be measured" in md
    assert "0 of 2" in md
    assert "0.0%" in md
