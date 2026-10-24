"""Batch evaluation: accuracy, precision, recall, F1, ROC-AUC, PR-AUC + 95% bootstrap CIs."""

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from data.dataset import ForgeryDataset
from model_factory import load_model_from_checkpoint


def bootstrap_ci(y_true, y_pred, y_prob, metric_fn, n_boot=2000, ci=0.95, rng=None):
    """Compute a bootstrap confidence interval for a metric."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(metric_fn(yt, yp, ypr))
        except Exception:
            continue
    if not scores:
        return (float("nan"), float("nan"))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(scores, 100 * alpha))
    hi = float(np.percentile(scores, 100 * (1 - alpha)))
    return (lo, hi)


def _acc(yt, yp, _):    return accuracy_score(yt, yp)
def _prec(yt, yp, _):   return precision_score(yt, yp, zero_division=0)
def _rec(yt, yp, _):    return recall_score(yt, yp, zero_division=0)
def _f1(yt, yp, _):     return f1_score(yt, yp, average="macro", zero_division=0)
def _roc(yt, _, ypr):   return roc_auc_score(yt, ypr)
def _pr(yt, _, ypr):    return average_precision_score(yt, ypr)


def tune_threshold(y_true, y_prob, lo: float = 0.01, hi: float = 0.99,
                   step: float = 0.01) -> float:
    """Probability cut maximising macro F1 on validation predictions."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def predict(model, loader, device, tta: bool = False):
    """Return (y_true, y_prob, paths) in dataset order; TTA averages the h-flip."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            probs = torch.softmax(model(inputs), dim=1)[:, 1]
            if tta:
                flipped = torch.softmax(model(torch.flip(inputs, dims=[3])), dim=1)[:, 1]
                probs = (probs + flipped) / 2
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    paths = [s[0] for s in loader.dataset.samples]
    return np.array(all_labels), np.array(all_probs), paths


def save_predictions(path: str, paths, y_true, y_prob) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, paths=np.array(paths), labels=np.asarray(y_true), probs=np.asarray(y_prob))


def is_degenerate(pred) -> bool:
    return len(np.unique(pred)) == 1


def macro_f1_at(y_true, y_prob, threshold: float) -> float:
    """Macro F1 after cutting probabilities at `threshold`."""
    return float(f1_score(y_true, (y_prob >= threshold).astype(int),
                          average="macro", zero_division=0))


def tune_tta_and_no_tta_thresholds(tta_true, tta_prob, no_tta_true, no_tta_prob):
    """Return (decision_threshold, threshold_no_tta), each tuned on its own distribution."""
    return (tune_threshold(tta_true, tta_prob),
            tune_threshold(no_tta_true, no_tta_prob))


def evaluate(checkpoint_path: str, data_dir: str, split: str = "test",
             batch_size: int = 16, num_workers: int = 0, n_boot: int = 2000,
             ela_quality: int = 75, tta: bool = False,
             threshold: float | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    arch = cfg.get("arch", "efficientnet_b0")
    input_mode = cfg.get("input_mode", "rgb_ela")
    run_name = cfg.get("run_name") or os.path.splitext(os.path.basename(checkpoint_path))[0]

    model = load_model_from_checkpoint(ckpt, device)
    model.eval()
    print(f"Loaded {arch}/{input_mode} from {checkpoint_path} "
          f"(val_f1={ckpt.get('val_f1', '?')})")

    def loader_for(s):
        ds = ForgeryDataset(
            data_dir, split=s, image_size=cfg.get("image_size", 224),
            input_mode=input_mode, ela_quality=ela_quality,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=False)

    target_loader = loader_for(split)
    y_true, y_prob, paths = predict(model, target_loader, device, tta=tta)

    tuned = threshold is None
    if tuned or tta:
        val_loader = target_loader if split == "val" else loader_for("val")

    if tta:
        _, y_prob_no_tta, _ = predict(model, target_loader, device, tta=False)
        if split == "val":
            nt_true, nt_prob = y_true, y_prob_no_tta
        else:
            nt_true, nt_prob, _ = predict(model, val_loader, device, tta=False)
    else:
        y_prob_no_tta = y_prob

    # Each variant is judged by a threshold tuned on its own validation probabilities.
    if tuned:
        if split == "val":
            val_true, val_prob = y_true, y_prob
        else:
            val_true, val_prob, _ = predict(model, val_loader, device, tta=tta)
        if tta:
            threshold, threshold_no_tta = tune_tta_and_no_tta_thresholds(
                val_true, val_prob, nt_true, nt_prob)
        else:
            threshold = threshold_no_tta = tune_threshold(val_true, val_prob)
    elif tta:
        threshold_no_tta = tune_threshold(nt_true, nt_prob)
    else:
        threshold_no_tta = threshold

    y_pred = (y_prob >= threshold).astype(int)
    pred_path = os.path.join("predictions", f"{run_name}_{split}.npz")
    save_predictions(pred_path, paths, y_true, y_prob)

    metrics = {}
    rng = np.random.default_rng(0)

    for name, fn in [("accuracy", _acc), ("precision", _prec), ("recall", _rec),
                     ("f1_macro", _f1), ("roc_auc", _roc), ("pr_auc", _pr)]:
        try:
            val = fn(y_true, y_pred, y_prob)
        except Exception:
            val = float("nan")
        lo, hi = bootstrap_ci(y_true, y_pred, y_prob, fn, n_boot=n_boot, rng=rng)
        metrics[name] = {"value": round(val, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    f1_argmax = macro_f1_at(y_true, y_prob, 0.5)
    f1_no_tta = macro_f1_at(y_true, y_prob_no_tta, threshold_no_tta)
    degenerate = is_degenerate(y_pred)

    print(f"\n{'Metric':<15} {'Value':>8}  {'95% CI':>18}")
    print("-" * 45)
    for name, m in metrics.items():
        print(f"{name:<15} {m['value']:>8.4f}  [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]")
    print(f"\nthreshold={threshold:.2f}  tta={tta}  f1_argmax={f1_argmax:.4f}  "
          f"f1_tuned={metrics['f1_macro']['value']:.4f}  "
          f"f1_tuned_no_tta={f1_no_tta:.4f}  degenerate={degenerate}")
    print(f"Predictions written to {pred_path}")

    result = {
        "checkpoint": checkpoint_path,
        "arch": arch,
        "input_mode": input_mode,
        "seed": cfg.get("seed", "?"),
        "split": split,
        "n_samples": len(y_true),
        "metrics": metrics,
        "tta": tta,
        "decision_threshold": round(float(threshold), 4),
        "f1_macro_argmax": round(f1_argmax, 4),
        "f1_macro_tuned": metrics["f1_macro"]["value"],
        "f1_macro_tuned_no_tta": round(f1_no_tta, 4),
        "degenerate": degenerate,
    }
    return result


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained model on a test split")
    p.add_argument("checkpoint", help="Path to model .pth checkpoint")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--ela-quality", type=int, default=75)
    p.add_argument("--tta", action="store_true",
                   help="Average probabilities over the image and its horizontal flip")
    p.add_argument("--threshold", type=float, default=None,
                   help="Fixed decision threshold (default: tune on validation)")
    p.add_argument("--out-json", type=str, default=None, help="Write results to JSON file")
    args = p.parse_args()

    result = evaluate(
        args.checkpoint, args.data_dir, split=args.split,
        batch_size=args.batch_size, num_workers=args.num_workers,
        n_boot=args.n_boot, ela_quality=args.ela_quality,
        tta=args.tta, threshold=args.threshold,
    )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to {args.out_json}")


if __name__ == "__main__":
    main()
