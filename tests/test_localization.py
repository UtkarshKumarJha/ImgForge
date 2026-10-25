import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from data.dataset import ForgeryDataset
from model_factory import build_model
from gradcam import GradCAM, get_gradcam_target_layer
from analysis.localization import (
    analyse, cam_to_mask, choose_threshold, coverage, iou_dice, render_markdown,
)


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
    eff = build_model("efficientnet_b0", "rgb_ela").eval()
    assert get_gradcam_target_layer(eff, "efficientnet_b0") is eff.features[-1]
    assert get_gradcam_target_layer(eff) is eff.features[-1]

    res = build_model("resnet18_bilstm", "rgb_ela").eval()
    assert get_gradcam_target_layer(res, "resnet18_bilstm") is res.backbone[-1][-1]
    assert get_gradcam_target_layer(res) is res.backbone[-1][-1]

    # The contract behind the Swin refusal: the chosen layer keeps (B, C, H, W).
    for model, layer, channels in ((eff, eff.features[-1], 1280),
                                   (res, res.backbone[-1][-1], 512)):
        captured = {}
        handle = layer.register_forward_hook(
            lambda _m, _i, out: captured.setdefault("out", out))
        with torch.no_grad():
            model(torch.randn(1, 4, 224, 224))
        handle.remove()
        out = captured["out"]
        assert out.dim() == 4 and out.shape[1] == channels
        assert out.shape[2] > 1 and out.shape[3] > 1


@pytest.mark.parametrize("arch", ["swin_tiny", "mlp"])
def test_gradcam_target_layer_refuses_arches_without_a_spatial_conv_trunk(arch):
    model = build_model(arch, "rgb_ela")
    for call in (lambda: get_gradcam_target_layer(model, arch),
                 lambda: get_gradcam_target_layer(model)):
        with pytest.raises(ValueError) as exc:
            call()
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


def test_cli_refuses_split_val_because_no_held_out_split_remains(tmp_path):
    out = tmp_path / "localization.md"
    r = subprocess.run(
        [sys.executable, "-m", "analysis.localization",
         "--ckpt", str(tmp_path / "never_loaded.pth"),
         "--data-dir", str(tmp_path), "--masks-dir", str(tmp_path / "masks"),
         "--split", "val", "--out", str(out)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )

    assert r.returncode == 1
    assert "no held-out split" in r.stderr
    assert not out.exists()


def _write_split(data_dir, split, rows):
    with open(data_dir / f"{split}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)


def _structured_image(path, rng):
    base = (np.linspace(0, 255, 224)[None, :, None]
            * np.linspace(1.0, 0.2, 224)[:, None, None])
    pixels = np.clip(np.repeat(base, 3, axis=2) + rng.normal(0, 12, (224, 224, 3)),
                     0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path, quality=90)
    return str(path)


def test_analyse_takes_the_threshold_from_val_and_scores_the_test_split(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    data = tmp_path / "data"
    masks = tmp_path / "masks"
    data.mkdir()
    masks.mkdir()

    val_t = _structured_image(data / "val_t.jpg", rng)
    val_a = _structured_image(data / "val_a.jpg", rng)
    t1 = _structured_image(data / "test_t1.jpg", rng)
    t2 = _structured_image(data / "test_t2.jpg", rng)
    auth = _structured_image(data / "auth.jpg", rng)
    ghost = str(data / "ghost_missing.jpg")     # CSV row + mask exist, image does not

    _write_split(data, "val", [(val_t, 1), (val_a, 0)])
    _write_split(data, "test", [(t1, 1), (t2, 1), (ghost, 1), (auth, 0)])

    model = build_model("efficientnet_b0", "rgb_ela", fusion="concat").eval()
    ckpt = tmp_path / "model.pth"
    torch.save({"cfg": {"arch": "efficientnet_b0", "input_mode": "rgb_ela",
                        "fusion": "concat", "image_size": 224},
                "model_state_dict": model.state_dict()}, ckpt)

    gradcam = GradCAM(model, get_gradcam_target_layer(model, "efficientnet_b0"))
    plan = {"val": [(val_t, 0.9)], "test": [(t1, 0.2), (t2, 0.2)]}
    cams224, mask_pixels = {}, {}
    for split, items in plan.items():
        dataset = ForgeryDataset(str(data), split=split, image_size=224,
                                 input_mode="rgb_ela")
        index_of = {path: i for i, (path, _) in enumerate(dataset.samples)}
        for path, cut in items:
            tensor, _ = dataset[index_of[path]]
            cam = cv2.resize(gradcam.generate(tensor.unsqueeze(0)), (224, 224))
            cams224[path] = cam
            mask_pixels[path] = cam >= cut
            Image.fromarray((mask_pixels[path] * 255).astype(np.uint8)).save(
                masks / f"{Path(path).stem}.png")
    # A non-empty mask for the missing image, so it is skipped for the image, not the mask.
    Image.fromarray((mask_pixels[t1] * 255).astype(np.uint8)).save(
        masks / "ghost_missing.png")

    result = analyse(str(ckpt), str(data), str(masks), "test",
                     grid=np.array([0.2, 0.5, 0.9]))

    assert result["reason"] is None
    assert result["threshold"] == 0.9          # val prefers 0.9
    assert result["val_mean_iou"] == 1.0
    assert result["n_scored"] == 2
    assert result["n_skipped_unreadable_images"] == 1

    # Test masks are {cam >= 0.2}, so the val threshold scores below 1 here: a mutant
    # choosing the threshold on this split would return 0.2 and a mean IoU of 1.0.
    expected_iou = np.mean([(cams224[p] >= 0.9).sum() / (cams224[p] >= 0.2).sum()
                            for p in (t1, t2)])
    expected_dice = np.mean([2 * (cams224[p] >= 0.9).sum()
                             / ((cams224[p] >= 0.9).sum() + (cams224[p] >= 0.2).sum())
                             for p in (t1, t2)])
    assert result["mean_iou"] == pytest.approx(expected_iou)
    assert result["mean_dice"] == pytest.approx(expected_dice)
    assert 0.0 < result["mean_iou"] < 1.0
    # Area reference points: an all-forged mask would score exactly mean_mask_area.
    assert result["mean_pred_area"] == pytest.approx(
        np.mean([(cams224[p] >= 0.9).mean() for p in (t1, t2)]))
    assert result["mean_mask_area"] == pytest.approx(
        np.mean([(cams224[p] >= 0.2).mean() for p in (t1, t2)]))

    md = render_markdown(result)
    assert "Validation calibration" in md
    assert "Threshold chosen: 0.90" in md
    assert "1 unreadable images" in md
