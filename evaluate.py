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
from model_factory import build_model


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


def evaluate(checkpoint_path: str, data_dir: str, split: str = "test",
             batch_size: int = 16, num_workers: int = 0, n_boot: int = 2000,
             ela_quality: int = 75):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    arch = cfg.get("arch", "efficientnet_b0")
    input_mode = cfg.get("input_mode", "rgb_ela")

    model = build_model(arch, input_mode)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded {arch}/{input_mode} from {checkpoint_path} "
          f"(val_f1={ckpt.get('val_f1', '?')})")

    ds = ForgeryDataset(
        data_dir, split=split, image_size=cfg.get("image_size", 224),
        input_mode=input_mode, ela_quality=ela_quality,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

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

    print(f"\n{'Metric':<15} {'Value':>8}  {'95% CI':>18}")
    print("-" * 45)
    for name, m in metrics.items():
        print(f"{name:<15} {m['value']:>8.4f}  [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]")

    result = {
        "checkpoint": checkpoint_path,
        "arch": arch,
        "input_mode": input_mode,
        "seed": cfg.get("seed", "?"),
        "split": split,
        "n_samples": len(y_true),
        "metrics": metrics,
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
    p.add_argument("--out-json", type=str, default=None, help="Write results to JSON file")
    args = p.parse_args()

    result = evaluate(
        args.checkpoint, args.data_dir, split=args.split,
        batch_size=args.batch_size, num_workers=args.num_workers,
        n_boot=args.n_boot, ela_quality=args.ela_quality,
    )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to {args.out_json}")


if __name__ == "__main__":
    main()
