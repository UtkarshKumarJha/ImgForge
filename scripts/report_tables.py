"""Regenerate docs/results/<arch>.md from the sweep's experiment_log.csv.

Only the declared configuration (no deviation suffix in run_name) feeds the
tables, so retry runs such as `efficientnet_b0_ela_p320_s1` never leak in.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

ARCHS = ["efficientnet_b0", "resnet18_bilstm", "swin_tiny", "mlp"]
MODES = ["rgb", "ela", "rgb_ela"]
# From the S0 gate (split_fingerprint of the canonical split); the interim report's
# 1,182 / 927 / 255 figures were wrong and must not come back.
SPLIT = ("1,181 test images (878 authentic / 303 tampered); split fingerprint "
         "`79e78eb230223f2775dbcd1422b35d14e775639ed049258621df5555ce225524`")
METRICS = ["f1_macro_tuned", "accuracy", "precision", "recall",
           "f1_macro_argmax", "f1_macro_tuned_no_tta", "roc_auc", "pr_auc"]


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def canonical(rows):
    """(arch, mode) -> [row], for rows whose run_name is exactly the declared one."""
    grouped = {}
    for row in rows:
        arch, mode, seed = row["arch"], row["input_mode"], row["seed"]
        expected = f"{arch}_{mode}_s{seed}"
        if row.get("run_name") and row["run_name"] != expected:
            continue
        grouped.setdefault((arch, mode), []).append(row)
    return grouped


def mean_sd(rows, metric):
    vals = [float(r[metric]) for r in rows if r.get(metric) not in ("", None)]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def table(arch, grouped):
    lines = [f"### {arch}", "",
             "| Input mode | Seeds | macro-F1 (tuned, TTA) | macro-F1 (argmax 0.5) | "
             "macro-F1 (no TTA) | Accuracy | ROC-AUC |",
             "|---|---:|---|---|---|---|---|"]
    for mode in MODES:
        rows = grouped.get((arch, mode))
        if not rows:
            lines.append(f"| {mode} | 0 | — | — | — | — | — |")
            continue
        cells = []
        for metric in ("f1_macro_tuned", "f1_macro_argmax", "f1_macro_tuned_no_tta",
                       "accuracy", "roc_auc"):
            m, sd = mean_sd(rows, metric)
            cells.append(f"{m:.4f} ± {sd:.4f}")
        lines.append(f"| {mode} | {len(rows)} | " + " | ".join(cells) + " |")
    return lines


def ci_cell(row):
    """The log stores `f1_macro_ci` as the "[lo, hi]" string evaluate.py printed."""
    parts = [p.strip() for p in (row.get("f1_macro_ci") or "").strip("[]").split(",")]
    if len(parts) != 2:
        return "—"
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        return "—"
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "—"
    return f"{lo:.3f}-{hi:.3f}"


def per_seed(arch, grouped):
    """Each seed's reported number with its own 95% bootstrap CI, so spread is visible."""
    lines = [f"### {arch} - per seed (val-tuned threshold, TTA on)", "",
             "| Input mode | Seed | macro-F1 | 95% bootstrap CI |",
             "|---|---:|---:|---|"]
    for mode in MODES:
        rows = sorted(grouped.get((arch, mode)) or [], key=lambda r: int(r["seed"]))
        for row in rows:
            lines.append(f"| {mode} | {row['seed']} | "
                         f"{float(row['f1_macro_tuned']):.4f} | {ci_cell(row)} |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", nargs="+", required=True,
                   help="experiment_log.csv from each session (merged)")
    p.add_argument("--out-dir", default="docs/results")
    p.add_argument("--retry-log", nargs="*", default=[],
                   help="logs holding documented retries (reported separately)")
    args = p.parse_args()

    grouped = canonical(load_rows(args.log))
    retry = load_rows(args.retry_log)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for arch in ARCHS:
        lines = [f"# {arch} — CASIA 2.0 sweep", "",
                 "Protocol: 224 px, 60-epoch cap with early stopping (patience 10), "
                 "batch 16, dual-stream stem, mode-aware augmentation, "
                 "discriminative LRs. Reported number is the **val-tuned threshold, "
                 "TTA-on** test macro-F1 across seeds (mean ± sd); argmax-0.5 and "
                 "no-TTA are shown for comparison.", "",
                 f"Measured on the held-out test split: {SPLIT}.", ""]
        lines += table(arch, grouped)
        lines += [""] + per_seed(arch, grouped)

        retry_rows = [r for r in retry
                      if r["arch"] == arch and r["run_name"] != f"{arch}_{r['input_mode']}_s{r['seed']}"]
        if retry_rows:
            lines += ["", "### Documented retry", "",
                      "| Run | Resolution | macro-F1 (tuned, TTA) | Accuracy |",
                      "|---|---:|---|---|"]
            for r in sorted(retry_rows, key=lambda r: r["run_name"]):
                tag = r["run_name"].split("_s")[0].split("_")
                res = next((t[1:] for t in tag if t.startswith("p") and t[1:].isdigit()), "224")
                lines.append(f"| {r['run_name']} | {res} | {float(r['f1_macro_tuned']):.4f} "
                             f"| {float(r['accuracy']):.4f} |")
            m, sd = mean_sd(retry_rows, "f1_macro_tuned")
            lines.append(f"| **retry mean** | | **{m:.4f} ± {sd:.4f}** | |")

        (out / f"{arch}.md").write_text("\n".join(lines) + "\n")
        print(f"wrote {out / (arch + '.md')}")


if __name__ == "__main__":
    main()
