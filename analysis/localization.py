"""Grad-CAM / ground-truth mask agreement (IoU, Dice), threshold chosen on val masks."""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from data.dataset import ForgeryDataset
from gradcam import GradCAM, get_gradcam_target_layer
from model_factory import build_from_checkpoint

DEFAULT_GRID = np.arange(0.05, 1.0, 0.05)
MASK_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def cam_to_mask(cam: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask at the CAM's own resolution; callers resize to the gt shape first."""
    return np.asarray(cam) >= threshold


def iou_dice(pred_mask, gt_mask) -> tuple[float, float]:
    """IoU and Dice against the ground truth; empty ground truth scores (0.0, 0.0)."""
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if not gt.any():
        return 0.0, 0.0
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    total = float(pred.sum() + gt.sum())
    return (inter / union if union else 0.0,
            2.0 * inter / total if total else 0.0)


def choose_threshold(cams, gts, grid) -> tuple[float | None, float]:
    """Threshold maximising mean IoU over pairs with foreground GT; ties pick the smallest."""
    best_threshold, best_iou = None, 0.0
    for threshold in np.sort(np.asarray(grid, dtype=float)):
        ious = [
            iou_dice(cam_to_mask(_resize_cam(cam, gt.shape), threshold), gt)[0]
            for cam, gt in zip(cams, gts)
            if np.asarray(gt).any()
        ]
        if not ious:
            continue
        mean_iou = float(np.mean(ious))
        if best_threshold is None or mean_iou > best_iou:
            best_threshold, best_iou = float(threshold), mean_iou
    return best_threshold, best_iou


def coverage(tampered_paths, mask_paths) -> float:
    """Fraction of tampered images with a mask whose filename stem matches exactly."""
    tampered_paths = list(tampered_paths)
    if not tampered_paths:
        return 0.0
    stems = {_stem(p) for p in mask_paths}
    return sum(_stem(p) in stems for p in tampered_paths) / len(tampered_paths)


def _stem(path) -> str:
    return Path(path).stem.lower()


def _resize_cam(cam: np.ndarray, shape) -> np.ndarray:
    return cv2.resize(np.asarray(cam, dtype=np.float32), (shape[1], shape[0]))


def _split_rows(data_dir: str, split: str) -> list[tuple[str, int]]:
    with open(Path(data_dir) / f"{split}.csv", newline="") as f:
        return [(row["path"], int(row["label"])) for row in csv.DictReader(f)]


def _mask_index(masks_dir: str) -> dict[str, Path]:
    """Map lower-case filename stem to mask path; duplicates resolve deterministically."""
    index: dict[str, Path] = {}
    for path in sorted(Path(masks_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in MASK_SUFFIXES:
            index.setdefault(_stem(path), path)
    return index


def _load_mask(path: Path) -> np.ndarray | None:
    """True where the mask marks forgery; None when the file cannot be decoded."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if mask is None else mask > 127


def _dataset(data_dir, split, cfg, input_mode):
    return ForgeryDataset(
        data_dir, split=split, image_size=cfg.get("image_size", 224),
        input_mode=input_mode,
    )


def _cams_and_masks(model, arch, dataset, image_paths, mask_index):
    """CAMs paired with their masks, skipping empty and unreadable masks."""
    gradcam = GradCAM(model, get_gradcam_target_layer(model, arch))
    index_of = {path: i for i, (path, _) in enumerate(dataset.samples)}
    cams, gts, n_empty, n_unreadable = [], [], 0, 0
    for path in image_paths:
        mask = _load_mask(mask_index[_stem(path)])
        if mask is None:
            n_unreadable += 1
            continue
        if not mask.any():
            n_empty += 1
            continue
        tensor, _ = dataset[index_of[path]]
        cams.append(gradcam.generate(tensor.unsqueeze(0)))
        gts.append(mask)
    return cams, gts, n_empty, n_unreadable


def analyse(ckpt_path: str, data_dir: str = "data", masks_dir: str = "data/masks",
            split: str = "test", grid=DEFAULT_GRID) -> dict:
    """Measure localization on `split`; nothing is loaded while coverage is still zero."""
    mask_index = _mask_index(masks_dir)
    tampered = [p for p, label in _split_rows(data_dir, split) if label == 1]
    matched = [p for p in tampered if _stem(p) in mask_index]
    result = {
        "ckpt": ckpt_path, "masks_dir": masks_dir, "split": split,
        "arch": None, "input_mode": None,
        "n_tampered": len(tampered), "n_matched": len(matched),
        "coverage": len(matched) / len(tampered) if tampered else 0.0,
        "val_coverage": None, "threshold": None, "val_mean_iou": None,
        "mean_iou": None, "mean_dice": None,
        "n_scored": 0, "n_skipped_empty": 0, "n_skipped_unreadable": 0,
        "reason": None,
    }

    if not tampered:
        result["reason"] = f"the {split} split has no tampered images"
        return result
    if not matched:
        result["reason"] = (f"no file under {masks_dir!r} has a filename stem matching "
                            f"a tampered {split} image")
        return result

    val_tampered = (tampered if split == "val"
                    else [p for p, label in _split_rows(data_dir, "val") if label == 1])
    val_matched = [p for p in val_tampered if _stem(p) in mask_index]
    result["val_coverage"] = (len(val_matched) / len(val_tampered) if val_tampered else 0.0)
    if not val_matched:
        result["reason"] = ("no file under the masks directory has a filename stem "
                            "matching a tampered val image, so the CAM threshold cannot "
                            "be chosen without touching the reported split")
        return result

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    arch = cfg.get("arch", "efficientnet_b0")
    input_mode = cfg.get("input_mode", "rgb_ela")
    result["arch"], result["input_mode"] = arch, input_mode

    model = build_from_checkpoint(ckpt)
    get_gradcam_target_layer(model, arch)   # refuses unsupported arches before any CAM work
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = _dataset(data_dir, split, cfg, input_mode)
    val_dataset = dataset if split == "val" else _dataset(data_dir, "val", cfg, input_mode)

    val_cams, val_gts, _, _ = _cams_and_masks(model, arch, val_dataset, val_matched, mask_index)
    threshold, val_mean_iou = choose_threshold(val_cams, val_gts, grid)
    if threshold is None:
        result["reason"] = "none of the matching val masks contains a forgery region"
        return result
    result["threshold"], result["val_mean_iou"] = threshold, val_mean_iou

    cams, gts, n_empty, n_unreadable = _cams_and_masks(
        model, arch, dataset, matched, mask_index)
    result["n_skipped_empty"], result["n_skipped_unreadable"] = n_empty, n_unreadable
    if not gts:
        result["reason"] = f"every matching {split} mask is empty or unreadable"
        return result

    scores = [iou_dice(cam_to_mask(_resize_cam(cam, gt.shape), threshold), gt)
              for cam, gt in zip(cams, gts)]
    result["n_scored"] = len(scores)
    result["mean_iou"] = float(np.mean([s[0] for s in scores]))
    result["mean_dice"] = float(np.mean([s[1] for s in scores]))
    return result


def render_markdown(result: dict) -> str:
    lines = [
        "# Localization: Grad-CAM / forgery-mask agreement",
        "",
        "Generated by `python -m analysis.localization`.",
        "",
        f"Checkpoint: `{result['ckpt']}`.",
        f"Masks: `{result['masks_dir']}`, matched to tampered images by exact, "
        "case-insensitive filename stem only. No fuzzy or suffix matching is used, "
        "so a mask folder that does not share the image files' stems contributes nothing.",
    ]
    if result["arch"]:
        lines.append(f"Model: {result['arch']} / {result['input_mode']}.")
    lines.append("")

    if result["reason"]:
        lines += [
            f"**Localization could not be measured on the {result['split']} split.** "
            f"{result['reason']}.",
            "",
            f"Coverage: {result['n_matched']} of {result['n_tampered']} tampered "
            f"{result['split']} images had a matching mask "
            f"({100 * result['coverage']:.1f}%).",
            "",
            "No IoU or Dice is reported, because a number computed from unmatched files "
            "would not describe this model.",
            "",
        ]
        return "\n".join(lines)

    lines += [
        f"Threshold on the Grad-CAM value in [0, 1]: {result['threshold']:.2f}, chosen "
        f"on the val split's masks only (mean IoU {result['val_mean_iou']:.4f}).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Coverage of tampered {result['split']} images | "
        f"{100 * result['coverage']:.1f}% "
        f"({result['n_matched']}/{result['n_tampered']}) |",
        f"| Images scored | {result['n_scored']} |",
        f"| Mean IoU | {result['mean_iou']:.4f} |",
        f"| Mean Dice | {result['mean_dice']:.4f} |",
        "",
        f"Skipped: {result['n_tampered'] - result['n_matched']} tampered images without a "
        f"mask, {result['n_skipped_empty']} masks with no forgery pixels, "
        f"{result['n_skipped_unreadable']} unreadable masks.",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description="Grad-CAM / ground-truth mask agreement (IoU, Dice)")
    p.add_argument("--ckpt", required=True, help="Path to a trained model .pth checkpoint")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--masks-dir", default="data/masks",
                   help="Folder of ground-truth forgery masks (searched recursively)")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="docs/results/localization.md")
    args = p.parse_args()

    try:
        result = analyse(args.ckpt, args.data_dir, args.masks_dir, args.split)
    except ValueError as e:
        raise SystemExit(str(e))

    print(f"tampered={result['n_tampered']}  matched={result['n_matched']}  "
          f"coverage={result['coverage']:.3f}  scored={result['n_scored']}")
    if result["reason"]:
        print(f"Localization could not be measured: {result['reason']}")
    else:
        print(f"threshold={result['threshold']:.2f} (val)  "
              f"mean_iou={result['mean_iou']:.4f}  mean_dice={result['mean_dice']:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(render_markdown(result))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
