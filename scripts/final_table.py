"""The report's single comparison table: every architecture on its best mode at one size.

Reads the sweep log and per-image predictions of runs named `{arch}_{mode}{suffix}_s{seed}`
and writes one markdown file with the cross-architecture table, per-seed CIs, 3-seed
ensembles, the JPEG-only slice and the Family B McNemar tests (Holm).

    python scripts/final_table.py --log merged/experiment_log.csv \
        --predictions merged/predictions --logs-dir merged/logs --suffix _p384 \
        --out docs/results/final_comparison.md
"""

import argparse
import os
import re
from itertools import combinations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from analysis.confound import jpeg_subset_f1
from analysis.mcnemar import analyse_family, load_predictions
from scripts.report_tables import ci_cell, load_rows, mean_sd

# Each architecture's best input mode in the 224 px grid (validation and test agree).
BEST_MODE = {"efficientnet_b0": "ela", "swin_tiny": "ela", "resnet18_bilstm": "ela", "mlp": "rgb"}
NAMES = {"efficientnet_b0": "EfficientNet-B0", "swin_tiny": "Swin-Tiny",
         "resnet18_bilstm": "ResNet-18 + BiLSTM", "mlp": "MLP"}
SEEDS = (1, 2, 3)


def run_name(arch: str, suffix: str, seed: int) -> str:
    return f"{arch}_{BEST_MODE[arch]}{suffix}_s{seed}"


def val_f1(logs_dir: str | None, name: str) -> float | None:
    """Best validation macro-F1 that train.py printed for this run."""
    path = os.path.join(logs_dir or "", f"{name}.log")
    if not logs_dir or not os.path.exists(path):
        return None
    found = re.findall(r"Best val F1: ([0-9.]+)", open(path, errors="replace").read())
    return float(found[-1]) if found else None


def ensemble(npz_paths: list[str]) -> dict:
    """Mean probability over seeds, cut at the mean of their val-tuned thresholds."""
    runs = [np.load(p) for p in npz_paths]
    paths = [str(p) for p in runs[0]["paths"]]
    for r in runs[1:]:
        assert [str(p) for p in r["paths"]] == paths, "seeds disagree on image order"
    probs = np.mean([r["probs"] for r in runs], axis=0)
    thr = float(np.mean([float(r["threshold"]) for r in runs]))
    y = np.asarray(runs[0]["labels"]).astype(int)
    pred = (probs >= thr).astype(int)
    return {"threshold": thr, "f1": f1_score(y, pred, average="macro"),
            "accuracy": accuracy_score(y, pred)}


def jpeg_only(npz_path: str) -> float | None:
    run = load_predictions(npz_path)
    kinds = ["jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "other" for p in run["paths"]]
    return jpeg_subset_f1(run["labels"], run["preds"], kinds)["f1"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", nargs="+", required=True)
    p.add_argument("--predictions", required=True, help="directory of <run>_test.npz")
    p.add_argument("--logs-dir", default=None, help="per-run training logs (val F1)")
    p.add_argument("--suffix", default="", help="deviation tag of the runs, e.g. _p384")
    p.add_argument("--out", default="docs/results/final_comparison.md")
    a = p.parse_args()

    rows = {r["run_name"]: r for r in load_rows(a.log)}
    npz = lambda name: os.path.join(a.predictions, f"{name}_test.npz")
    runs = {arch: [rows[n] for n in (run_name(arch, a.suffix, s) for s in SEEDS) if n in rows]
            for arch in BEST_MODE}
    res = re.search(r"_p(\d+)", a.suffix)
    size = res.group(1) if res else "224"

    out = [f"# Final comparison — {size} px, each architecture on its best input mode", "",
           "Test split: 1,181 images (878 authentic / 303 tampered). Reported number: "
           "val-tuned threshold, TTA on; mean ± sd over seeds. Precision and recall are for "
           "the tampered class.", "",
           "| Model | Input | Seeds | Macro-F1 | Accuracy | Precision | Recall | ROC-AUC | PR-AUC |",
           "|---|---|---:|---|---|---|---|---|---|"]
    for arch, rs in runs.items():
        cells = []
        for m in ("f1_macro_tuned", "accuracy", "precision", "recall", "roc_auc", "pr_auc"):
            mean, sd = mean_sd(rs, m)
            cells.append("—" if mean is None else f"{mean:.4f} ± {sd:.4f}")
        out.append(f"| {NAMES[arch]} | {BEST_MODE[arch]} | {len(rs)} | " + " | ".join(cells) + " |")

    out += ["", "## Per seed", "",
            "| Model | Seed | Val macro-F1 | Test macro-F1 | 95% bootstrap CI | JPEG-only macro-F1 |",
            "|---|---:|---:|---:|---|---:|"]
    best = {}
    for arch, rs in runs.items():
        for r in sorted(rs, key=lambda r: int(r["seed"])):
            v = val_f1(a.logs_dir, r["run_name"])
            j = jpeg_only(npz(r["run_name"])) if os.path.exists(npz(r["run_name"])) else None
            out.append(f"| {NAMES[arch]} | {r['seed']} | {'—' if v is None else f'{v:.4f}'} | "
                       f"{float(r['f1_macro_tuned']):.4f} | {ci_cell(r)} | "
                       f"{'—' if j is None else f'{j:.4f}'} |")
            if v is not None and (arch not in best or v > best[arch][0]):
                best[arch] = (v, r["run_name"])

    out += ["", "## 3-seed ensembles (mean probability, mean of the members' val thresholds)", "",
            "| Model | Threshold | Macro-F1 | Accuracy |", "|---|---:|---:|---:|"]
    for arch, rs in runs.items():
        files = [npz(r["run_name"]) for r in rs]
        if len(files) == len(SEEDS) and all(map(os.path.exists, files)):
            e = ensemble(files)
            out.append(f"| {NAMES[arch]} | {e['threshold']:.3f} | {e['f1']:.4f} | {e['accuracy']:.4f} |")

    if len(best) == len(BEST_MODE):
        reps = {arch: best[arch][1] for arch in BEST_MODE}
        preds = {name: load_predictions(npz(name)) for name in reps.values()}
        family = analyse_family(list(combinations(reps.values(), 2)), preds)
        out += ["", "## McNemar, Family B (representative = best validation seed), Holm-corrected",
                "", "| Pair | b (A right, B wrong) | c (A wrong, B right) | p | p (Holm) |",
                "|---|---:|---:|---:|---:|"]
        for f in family:
            out.append(f"| {f['pair']} | {f['b']} | {f['c']} | {f['p']:.3g} | {f['p_holm']:.3g} |")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
