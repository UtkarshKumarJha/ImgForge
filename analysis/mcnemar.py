"""Exact McNemar tests with Holm correction over two pre-declared families."""

import argparse
import glob
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from model_factory import VALID_ARCHS, VALID_MODES

FAMILY_A_SIZE = len(VALID_ARCHS) * (len(VALID_MODES) * (len(VALID_MODES) - 1) // 2)
FAMILY_B_SIZE = len(VALID_ARCHS) * (len(VALID_ARCHS) - 1) // 2


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    p = binomtest(min(b, c), n, 0.5).pvalue
    return float(min(max(p, 0.0), 1.0))


def mcnemar_pair(y_true, pred_a, pred_b) -> tuple[int, int, float]:
    """Return (b, c, p): b = A correct and B wrong, c = A wrong and B correct."""
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError(
            f"y_true, pred_a and pred_b must have equal length "
            f"({len(y_true)}, {len(pred_a)}, {len(pred_b)})")
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    return b, c, mcnemar_exact(b, c)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, monotone and capped at 1."""
    m = len(pvals)
    adjusted = {}
    running = 0.0
    for i, key in enumerate(sorted(pvals, key=pvals.get)):
        running = max(running, (m - i) * pvals[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def parse_run_name(name: str) -> tuple[str, str]:
    """Split `{arch}_{mode}_s{seed}` using the longest known arch and mode names."""
    for arch in sorted(VALID_ARCHS, key=len, reverse=True):
        prefix = f"{arch}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        for mode in sorted(VALID_MODES, key=len, reverse=True):
            if rest == mode or rest.startswith(f"{mode}_"):
                return arch, mode
    raise ValueError(
        f"Cannot parse run name {name!r} against VALID_ARCHS and VALID_MODES")


def within_arch_pairs(run_names: list[str]) -> list[tuple[str, str]]:
    """All input-mode pairs inside each architecture (3 per arch, 12 total)."""
    by_arch: dict[str, dict[str, str]] = {}
    for name in run_names:
        arch, mode = parse_run_name(name)
        modes = by_arch.setdefault(arch, {})
        if mode in modes:
            raise ValueError(
                f"Duplicate (arch, input_mode) for {arch}/{mode}: "
                f"{modes[mode]!r} and {name!r}")
        modes[mode] = name
    return [
        (by_arch[arch][m1], by_arch[arch][m2])
        for arch in sorted(by_arch)
        for m1, m2 in combinations(sorted(by_arch[arch]), 2)
    ]


def cross_arch_pairs(best_run_per_arch: dict[str, str]) -> list[tuple[str, str]]:
    """All pairs between architecture representatives (6 for four archs)."""
    archs = sorted(best_run_per_arch)
    return [(best_run_per_arch[a], best_run_per_arch[b])
            for a, b in combinations(archs, 2)]


def _checkpoint_val_f1(path: str) -> float | None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    val_f1 = ckpt.get("val_f1")
    return None if val_f1 is None else float(val_f1)


def _discover_checkpoints(models_dir: str) -> dict:
    """Map (arch, mode, seed) to (run_name, path) for run checkpoints we understand."""
    found = {}
    for path in sorted(glob.glob(os.path.join(models_dir, "*_best.pth"))):
        run_name = Path(path).stem[: -len("_best")]
        parts = run_name.rsplit("_s", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        try:
            arch, mode = parse_run_name(run_name)
        except ValueError:
            continue
        found[(arch, mode, int(parts[1]))] = (run_name, path)
    return found


def select_best_runs(models_dir: str) -> tuple[dict, dict]:
    """Best checkpoint val_f1 per (arch, mode) and per arch; a NaN never wins."""
    found = _discover_checkpoints(models_dir)
    seeds = sorted({seed for _, _, seed in found})
    missing_runs = [
        (f"{arch}_{mode}_s{seed}",
         os.path.join(models_dir, f"{arch}_{mode}_s{seed}_best.pth"))
        for arch in sorted(VALID_ARCHS)
        for mode in sorted(VALID_MODES)
        for seed in seeds
        if (arch, mode, seed) not in found
    ]

    best_by_mode: dict[tuple[str, str], tuple[tuple[float, int], str]] = {}
    best_by_arch: dict[str, tuple[tuple[float, int], str]] = {}
    for (arch, mode, seed), (run_name, path) in sorted(found.items()):
        val_f1 = _checkpoint_val_f1(path)
        if val_f1 is None or not np.isfinite(val_f1):
            continue
        score = (val_f1, -seed)          # lower seed wins a tie
        if (arch, mode) not in best_by_mode or score > best_by_mode[(arch, mode)][0]:
            best_by_mode[(arch, mode)] = (score, run_name)
        if arch not in best_by_arch or score > best_by_arch[arch][0]:
            best_by_arch[arch] = (score, run_name)

    missing_cells = [
        (arch, mode)
        for arch in sorted(VALID_ARCHS)
        for mode in sorted(VALID_MODES)
        if (arch, mode) not in best_by_mode
    ]
    missing_archs = [arch for arch in sorted(VALID_ARCHS)
                     if arch not in best_by_arch]
    if missing_runs or missing_cells or missing_archs:
        lines = [f"Incomplete checkpoint set under {models_dir!r}:"]
        if missing_runs:
            lines.append("  missing runs:")
            lines += [f"    {run_name} ({path})"
                      for run_name, path in missing_runs]
        if missing_cells:
            lines.append("  missing (arch, input mode) cells: "
                         + ", ".join(f"{a}/{m}" for a, m in missing_cells))
        if missing_archs:
            lines.append("  missing architectures: " + ", ".join(missing_archs))
        raise ValueError("\n".join(lines))

    return (
        {key: value[1] for key, value in best_by_mode.items()},
        {arch: value[1] for arch, value in best_by_arch.items()},
    )


def load_predictions(npz_path: str, threshold: float = 0.5) -> dict:
    """Load one evaluate.py artefact and cut probabilities into hard predictions."""
    with np.load(npz_path) as d:
        paths = [str(p) for p in d["paths"]]
        labels = np.asarray(d["labels"]).astype(int)
        probs = np.asarray(d["probs"]).astype(float)
    if not (len(paths) == len(labels) == len(probs)):
        raise ValueError(
            f"{npz_path}: paths/labels/probs have different lengths "
            f"({len(paths)}, {len(labels)}, {len(probs)})")
    return {"paths": paths, "labels": labels,
            "preds": (probs >= threshold).astype(int)}


def align_runs(run_a: dict, run_b: dict) -> dict:
    """Intersect two runs by image path; never rely on row order or position."""
    a = dict(zip(run_a["paths"], zip(run_a["labels"], run_a["preds"])))
    b = dict(zip(run_b["paths"], zip(run_b["labels"], run_b["preds"])))
    common = sorted(a.keys() & b.keys())
    if not common:
        raise ValueError("The two prediction files share no image paths")
    if any(a[p][0] != b[p][0] for p in common):
        raise ValueError("Labels disagree on shared image paths")
    return {
        "y_true": np.array([a[p][0] for p in common]),
        "pred_a": np.array([a[p][1] for p in common]),
        "pred_b": np.array([b[p][1] for p in common]),
        "n": len(common),
        "n_a": len(run_a["paths"]),
        "n_b": len(run_b["paths"]),
    }


def analyse_family(pairs: list[tuple[str, str]], predictions: dict) -> list[dict]:
    """Run every pair, then Holm-correct the p-values within this family only."""
    rows = []
    for name_a, name_b in pairs:
        al = align_runs(predictions[name_a], predictions[name_b])
        b, c, p = mcnemar_pair(al["y_true"], al["pred_a"], al["pred_b"])
        rows.append({
            "pair": f"{name_a} vs {name_b}",
            "b": b, "c": c, "p": p,
            "n": al["n"], "n_a": al["n_a"], "n_b": al["n_b"],
        })
    adjusted = holm({row["pair"]: row["p"] for row in rows})
    for row in rows:
        row["p_holm"] = adjusted[row["pair"]]
    return rows


def parse_prediction_file(path: str) -> tuple[str, str]:
    """Split predictions/{run_name}_{split}.npz into (run_name, split)."""
    stem = Path(path).stem
    for split in ("test", "val"):
        suffix = f"_{split}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], split
    raise ValueError(
        f"{path}: expected a predictions/{{run_name}}_{{test,val}}.npz file")


def _fmt(x, spec: str = ".3g") -> str:
    return "n/a" if x is None else format(x, spec)


def render_markdown(families, models_dir: str, threshold: float) -> str:
    lines = [
        "# McNemar tests with Holm correction",
        "",
        "Generated by `python -m analysis.mcnemar`.",
        f"Decision threshold: {threshold:.2f}.",
        "Representatives are the best validation macro-F1 run per (arch, input mode) "
        f"for Family A and per architecture for Family B, read from each run's best "
        f"checkpoint under `{models_dir}` (the same `val_f1` early stopping used); "
        "every McNemar test is scored on the test split, so selection never uses the "
        "reported split.",
        "Predictions are aligned by image path before testing; `n` is the number of "
        "shared paths. Pairs whose path sets do not fully overlap are flagged below.",
        "Holm correction is applied separately within each pre-declared family; the "
        "families are never pooled.",
        "",
    ]
    for title, rows in families:
        lines += [
            f"## {title}",
            "",
            "_A_ is the left-hand run in each pair and _B_ the right-hand run; "
            "`b` counts images A gets right and B gets wrong, `c` counts images "
            "A gets wrong and B right.",
            "",
            "| Pair | n | b | c | p | p_holm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        if not rows:
            lines.append("| _none_ | | | | | |")
        for row in rows:
            lines.append(
                f"| `{row['pair']}` | {row['n']} | {row['b']} | {row['c']} "
                f"| {_fmt(row['p'])} | {_fmt(row['p_holm'])} |")
        reduced = [row for row in rows
                   if row["n"] < row["n_a"] or row["n"] < row["n_b"]]
        if reduced:
            lines += ["", "Reduced path overlap (compared / images in each run):"]
            lines += [f"- `{row['pair']}`: {row['n']} of {row['n_a']} / {row['n_b']}."
                      for row in reduced]
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description="Exact McNemar tests with Holm correction over two declared families")
    p.add_argument("--predictions", nargs="+", required=True,
                   help="One or more predictions/{run}_{split}.npz files")
    p.add_argument("--models-dir", default="models",
                   help="Directory holding {run_name}_best.pth checkpoints with val_f1")
    p.add_argument("--out", default="docs/results/mcnemar.md")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability cut for hard predictions (default: 0.5)")
    args = p.parse_args()

    try:
        best_by_mode, best_by_arch = select_best_runs(args.models_dir)
    except ValueError as e:
        raise SystemExit(str(e))

    family_a = within_arch_pairs(list(best_by_mode.values()))
    family_b = cross_arch_pairs(best_by_arch)
    if len(family_a) != FAMILY_A_SIZE or len(family_b) != FAMILY_B_SIZE:
        raise SystemExit(
            f"Expected {FAMILY_A_SIZE} within-architecture and {FAMILY_B_SIZE} "
            f"cross-architecture pairs, got {len(family_a)} and {len(family_b)}")

    needed = sorted({name for pair in family_a + family_b for name in pair})
    available = {}
    for path in args.predictions:
        run, split = parse_prediction_file(path)
        if split == "test" and run in needed:
            available.setdefault(run, path)
    missing = [name for name in needed if name not in available]
    if missing:
        raise SystemExit("Missing test predictions for: " + ", ".join(missing))

    predictions = {name: load_predictions(available[name], args.threshold)
                   for name in needed}
    families = [
        (f"Family A: within-architecture input modes ({len(family_a)} tests)",
         analyse_family(family_a, predictions)),
        (f"Family B: cross-architecture representatives ({len(family_b)} tests)",
         analyse_family(family_b, predictions)),
    ]

    for title, rows in families:
        print(f"\n{title}")
        for row in rows:
            print(f"  {row['pair']}  n={row['n']}  b={row['b']}  c={row['c']}  "
                  f"p={_fmt(row['p'])}  p_holm={_fmt(row['p_holm'])}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(render_markdown(families, args.models_dir, args.threshold))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
