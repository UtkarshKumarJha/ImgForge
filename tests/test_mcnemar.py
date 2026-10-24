import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from analysis.mcnemar import (
    align_runs,
    cross_arch_pairs,
    holm,
    load_predictions,
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
    runs = [f"{a}_{m}_s1" for a in ARCHS for m in MODES]
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


def test_within_arch_pairs_rejects_duplicate_cells():
    runs = [f"{a}_{m}_s{s}" for a in ARCHS for m in MODES for s in (1, 2)]
    try:
        within_arch_pairs(runs)
    except ValueError as e:
        assert "duplicate" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_alignment_is_by_path_not_row_order():
    a = {"paths": ["x", "y"], "labels": np.array([0, 1]), "preds": np.array([0, 1])}
    b = {"paths": ["y", "x"], "labels": np.array([1, 0]), "preds": np.array([0, 0])}
    al = align_runs(a, b)
    assert al["n"] == 2
    assert mcnemar_pair(al["y_true"], al["pred_a"], al["pred_b"])[:2] == (1, 0)


# --- checkpoint selection -------------------------------------------------

def _write_checkpoints(models_dir, values=None):
    """One tiny checkpoint per sweep run, named like train.py writes them."""
    models_dir.mkdir(exist_ok=True)
    values = values or {}
    for arch in ARCHS:
        for mode in MODES:
            for seed in (1, 2, 3):
                run = f"{arch}_{mode}_s{seed}"
                torch.save({"val_f1": values.get(run, 0.80)},
                           models_dir / f"{run}_best.pth")


def test_best_runs_use_checkpoint_val_f1(tmp_path):
    models = tmp_path / "models"
    _write_checkpoints(models, {
        "mlp_rgb_s1": 0.90, "mlp_rgb_s2": 0.95, "mlp_rgb_s3": 0.80,
        "mlp_ela_s1": 0.80, "mlp_ela_s2": 0.60, "mlp_ela_s3": 0.70,
        "mlp_rgb_ela_s1": 0.80, "mlp_rgb_ela_s2": 0.80, "mlp_rgb_ela_s3": 0.80,
    })

    by_mode, by_arch = select_best_runs(str(models))
    # seed means: s1=(.90+.80+.80)/3=.8333, s2=(.95+.60+.80)/3=.7833, s3=.7667
    # Family A must not mix seeds, so s2's strong rgb run cannot be used.
    assert by_mode[("mlp", "rgb")] == "mlp_rgb_s1"
    assert by_mode[("mlp", "ela")] == "mlp_ela_s1"
    assert by_mode[("mlp", "rgb_ela")] == "mlp_rgb_ela_s1"
    # Family B has no seed to hold fixed: it takes the single best run per arch.
    assert by_arch["mlp"] == "mlp_rgb_s2"


def test_family_a_breaks_equal_seed_means_with_the_lower_seed(tmp_path):
    models = tmp_path / "models"
    _write_checkpoints(models)          # every run 0.80, so all seed means tie

    by_mode, _ = select_best_runs(str(models))
    assert all(name.endswith("_s1") for name in by_mode.values())


def test_nan_val_f1_is_never_selected(tmp_path):
    models = tmp_path / "models"
    _write_checkpoints(models, {
        "mlp_rgb_s1": float("nan"), "mlp_rgb_s2": 0.70, "mlp_rgb_s3": 0.60,
    })

    by_mode, _ = select_best_runs(str(models))
    assert by_mode[("mlp", "rgb")] == "mlp_rgb_s2"


def test_missing_checkpoint_names_run_and_path(tmp_path):
    models = tmp_path / "models"
    _write_checkpoints(models)
    missing = models / "mlp_rgb_s2_best.pth"
    missing.unlink()

    try:
        select_best_runs(str(models))
    except ValueError as e:
        assert "mlp_rgb_s2" in str(e)
        assert str(missing) in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_incomplete_coverage_names_cells_and_archs(tmp_path):
    models = tmp_path / "models"
    nan = {f"efficientnet_b0_{m}_s{s}": float("nan")
           for m in MODES for s in (1, 2, 3)}
    _write_checkpoints(models, nan)

    try:
        select_best_runs(str(models))
    except ValueError as e:
        msg = str(e)
        assert "efficientnet_b0/ela" in msg
        assert "efficientnet_b0/rgb" in msg
        assert "efficientnet_b0/rgb_ela" in msg
        assert "missing architectures: efficientnet_b0" in msg
    else:
        raise AssertionError("expected ValueError")


# --- CLI end to end -------------------------------------------------------

LABELS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
WRONG = {
    "efficientnet_b0": set(),
    "mlp": {0, 1, 2},
    "resnet18_bilstm": {0, 1, 2, 3, 4},
    "swin_tiny": {0, 1, 5, 6},
}


def _arch_probs(arch):
    probs = []
    for i, label in enumerate(LABELS):
        pred = 1 - label if i in WRONG[arch] else label
        probs.append(0.9 if pred == 1 else 0.1)
    return probs


def _write_npz(path, paths, labels, probs, threshold=None, tta=False):
    np.savez(path, paths=np.array(paths), labels=np.array(labels),
             probs=np.array(probs),
             threshold=np.float64(threshold if threshold is not None else np.nan),
             tta=np.bool_(tta))


def test_load_predictions_uses_the_artefact_threshold_and_warns_on_fallback(
        tmp_path, capsys):
    p = tmp_path / "a.npz"
    _write_npz(p, ["x"], [1], [0.4], threshold=0.3)
    # 0.40 clears the run's own 0.30 cut but not 0.50.
    assert load_predictions(str(p))["preds"].tolist() == [1]
    assert load_predictions(str(p), threshold=0.5)["preds"].tolist() == [0]

    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, paths=["x"], labels=[1], probs=[0.4])
    assert load_predictions(str(legacy))["preds"].tolist() == [0]
    assert "no tuned threshold" in capsys.readouterr().out


def _family_rows(md, title):
    section = md.split(title, 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append((cells[0].strip("`"), float(cells[4]), float(cells[5])))
    return rows


def test_cli_reports_both_families_and_path_overlap(tmp_path):
    models = tmp_path / "models"
    _write_checkpoints(models)

    paths = [f"p{i}" for i in range(len(LABELS))]
    pred_files = []
    for arch in ARCHS:
        probs = _arch_probs(arch)
        for mode in MODES:
            for seed in (1, 2, 3):
                run = f"{arch}_{mode}_s{seed}"   # val ties break to seed 1
                p = tmp_path / f"{run}_test.npz"
                if run == "mlp_ela_s1":
                    _write_npz(p, paths[:-1], LABELS[:-1], probs[:-1], threshold=0.43)
                else:
                    _write_npz(p, paths, LABELS, probs, threshold=0.43)
                pred_files.append(str(p))

    out = tmp_path / "mcnemar.md"
    subprocess.run(
        [sys.executable, "-m", "analysis.mcnemar", "--predictions", *pred_files,
         "--models-dir", str(models), "--out", str(out)],
        cwd=Path(__file__).resolve().parents[1], check=True,
        capture_output=True, text=True,
    )

    md = out.read_text()
    assert "Family A" in md and "12 tests" in md
    assert "Family B" in md and "6 tests" in md
    assert sum(line.startswith("| `") for line in md.splitlines()) == 18
    assert "validation" in md and "test split" in md
    assert "`mlp_ela_s1`=0.43" in md      # the artefacts' own cuts, not 0.5
    assert "0.50" not in md
    assert md.count("`b` counts images A") == 2
    assert "Reduced path overlap" in md
    assert "`mlp_ela_s1 vs mlp_rgb_s1`: 9 of 9 / 10" in md

    # Each family's p_holm must come from its own Holm run, not a pooled 18-test one.
    family_b = _family_rows(md, "Family B:")
    assert any(p_holm < 1 for _, _, p_holm in family_b)
    for title, expected in (("Family A:", 12), ("Family B:", 6)):
        rows = _family_rows(md, title)
        assert len(rows) == expected
        adjusted = holm({pair: p for pair, p, _ in rows})
        for pair, _, p_holm in rows:
            assert abs(adjusted[pair] - p_holm) < 5e-4
