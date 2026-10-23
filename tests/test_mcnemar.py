import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from analysis.mcnemar import (
    align_runs,
    cross_arch_pairs,
    holm,
    mcnemar_exact,
    mcnemar_pair,
    parse_run_name,
    select_best_runs,
    within_arch_pairs,
)

ARCHS = ["efficientnet_b0", "mlp", "resnet18_bilstm", "swin_tiny"]
MODES = ["ela", "rgb", "rgb_ela"]


def test_exact_p_value_matches_binomial_for_known_counts():
    assert abs(mcnemar_exact(10, 2) - 0.038574) < 1e-5
    assert mcnemar_exact(5, 5) == 1.0


def test_pair_counts_discordant_cells_only():
    y = np.array([1, 1, 0, 0])
    a = np.array([1, 1, 0, 1])   # wrong on last
    b = np.array([1, 0, 0, 0])   # wrong on second
    bb, cc, _ = mcnemar_pair(y, a, b)
    assert (bb, cc) == (1, 1)


def test_holm_is_monotone_and_bounded():
    out = holm({"a": 0.001, "b": 0.02, "c": 0.04})
    assert out["a"] <= out["b"] <= out["c"]
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_within_arch_pairs_yields_twelve():
    runs = [f"{a}_{m}_s1" for a in ["efficientnet_b0", "mlp", "resnet18_bilstm", "swin_tiny"]
            for m in ["ela", "rgb", "rgb_ela"]]
    assert len(within_arch_pairs(runs)) == 12


# --- binding-decision coverage -------------------------------------------

def test_pair_b_means_a_correct_b_wrong():
    y = np.array([1, 1, 0, 0])
    a = np.array([1, 1, 0, 0])   # all correct
    b = np.array([1, 1, 0, 1])   # wrong on last
    assert mcnemar_pair(y, a, b)[:2] == (1, 0)


def test_pair_rejects_mismatched_lengths():
    try:
        mcnemar_pair(np.array([0, 1]), np.array([0]), np.array([0, 1]))
    except ValueError as e:
        assert "length" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_zero_discordant_pairs_are_not_significant():
    y = np.array([0, 1, 0, 1])
    assert mcnemar_pair(y, y.copy(), y.copy()) == (0, 0, 1.0)


def test_holm_matches_worked_example():
    out = holm({"a": 0.001, "b": 0.02, "c": 0.04})
    assert abs(out["a"] - 0.003) < 1e-12
    assert abs(out["b"] - 0.040) < 1e-12
    assert abs(out["c"] - 0.040) < 1e-12


def test_parse_run_name_handles_underscored_arch_and_modes():
    assert parse_run_name("resnet18_bilstm_rgb_ela_s1") == ("resnet18_bilstm", "rgb_ela")
    assert parse_run_name("efficientnet_b0_rgb_s2") == ("efficientnet_b0", "rgb")


def test_cross_arch_pairs_yields_six():
    best = {a: f"{a}_rgb_s1" for a in ARCHS}
    assert len(cross_arch_pairs(best)) == 6


def test_alignment_is_by_path_not_row_order():
    a = {"paths": ["x", "y"], "labels": np.array([0, 1]), "preds": np.array([0, 1])}
    b = {"paths": ["y", "x"], "labels": np.array([1, 0]), "preds": np.array([0, 0])}
    al = align_runs(a, b)
    assert al["n"] == 2
    assert mcnemar_pair(al["y_true"], al["pred_a"], al["pred_b"])[:2] == (1, 0)


def test_best_runs_use_validation_f1_not_test_f1(tmp_path):
    log = tmp_path / "log.csv"
    with open(log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "input_mode", "seed", "split", "f1_macro"])
        w.writerow(["mlp", "rgb", 1, "val", 0.90])
        w.writerow(["mlp", "rgb", 2, "val", 0.95])
        w.writerow(["mlp", "rgb", 3, "val", 0.80])
        w.writerow(["mlp", "rgb", 1, "test", 0.99])
        w.writerow(["mlp", "rgb", 2, "test", 0.10])
        # equal F1, higher seed listed first: the lower seed must still win
        w.writerow(["mlp", "ela", 2, "val", 0.80])
        w.writerow(["mlp", "ela", 1, "val", 0.80])

    by_mode, by_arch = select_best_runs(str(log))
    assert by_mode[("mlp", "rgb")] == "mlp_rgb_s2"
    assert by_mode[("mlp", "ela")] == "mlp_ela_s1"
    assert by_arch["mlp"] == "mlp_rgb_s2"


def _write_full_log(path):
    """36-run val log plus deliberately misleading test rows."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "input_mode", "seed", "split", "f1_macro"])
        for arch in ARCHS:
            for mode in MODES:
                for seed in (1, 2, 3):
                    w.writerow([arch, mode, seed, "val", 0.80])
                    w.writerow([arch, mode, seed, "test", 0.10])


def _write_npz(path, paths, labels, probs):
    np.savez(path, paths=np.array(paths), labels=np.array(labels),
             probs=np.array(probs))


def test_cli_reports_both_families_and_path_overlap(tmp_path):
    log = tmp_path / "experiment_log.csv"
    _write_full_log(log)

    labels = [0, 1, 0, 1, 0, 1]
    probs = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    paths = [f"p{i}" for i in range(6)]

    pred_files = []
    for arch in ARCHS:
        for mode in MODES:
            run = f"{arch}_{mode}_s1"     # val ties break to the lowest seed
            p = tmp_path / f"{run}_test.npz"
            if run == "mlp_ela_s1":
                _write_npz(p, paths[:-1], labels[:-1], probs[:-1])  # reduced overlap
            else:
                _write_npz(p, paths, labels, probs)
            pred_files.append(str(p))

    out = tmp_path / "mcnemar.md"
    subprocess.run(
        [sys.executable, "-m", "analysis.mcnemar", "--predictions", *pred_files,
         "--log", str(log), "--out", str(out)],
        cwd=Path(__file__).resolve().parents[1], check=True,
        capture_output=True, text=True,
    )

    md = out.read_text()
    assert "Family A" in md and "12 tests" in md
    assert "Family B" in md and "6 tests" in md
    assert sum(line.startswith("| `") for line in md.splitlines()) == 18
    assert "validation" in md and "test split" in md
    assert "Decision threshold: 0.50" in md
    assert "Reduced path overlap" in md
    assert "`mlp_ela_s1 vs mlp_rgb_s1`: 5 of 5 / 6" in md
