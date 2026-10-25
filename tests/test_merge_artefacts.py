import csv

from scripts.merge_artefacts import merge


def _persist(root, names, log_rows):
    for sub in ("models", "predictions"):
        (root / sub).mkdir(parents=True, exist_ok=True)
        for name in names:
            (root / sub / name).write_text("x")
    with open(root / "experiment_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_name", "arch"])
        w.writeheader()
        w.writerows(log_rows)


def test_merge_dedupes_files_and_excludes_retry_runs(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _persist(a, ["efficientnet_b0_ela_s1_best.pth", "efficientnet_b0_ela_p320_s1_best.pth"],
             [{"run_name": "efficientnet_b0_ela_s1", "arch": "efficientnet_b0"},
              {"run_name": "efficientnet_b0_ela_p320_s1", "arch": "efficientnet_b0"}])
    _persist(b, ["efficientnet_b0_ela_s1_best.pth", "resnet18_bilstm_rgb_s2_best.pth"],
             [{"run_name": "efficientnet_b0_ela_s1", "arch": "efficientnet_b0"},
              {"run_name": "resnet18_bilstm_rgb_s2", "arch": "resnet18_bilstm"}])

    out = tmp_path / "out"
    counts = merge([str(a), str(b)], str(out))

    # The retry is neither copied nor logged; the duplicate run is counted once.
    assert sorted(p.name for p in (out / "models").iterdir()) == [
        "efficientnet_b0_ela_s1_best.pth", "resnet18_bilstm_rgb_s2_best.pth"]
    assert counts["log_rows"] == 2
    rows = list(csv.DictReader(open(out / "experiment_log.csv")))
    assert [r["run_name"] for r in rows] == [
        "efficientnet_b0_ela_s1", "resnet18_bilstm_rgb_s2"]


def test_retries_can_be_kept_for_their_own_section(tmp_path):
    a = tmp_path / "a"
    _persist(a, ["efficientnet_b0_ela_p320_s1_best.pth"],
             [{"run_name": "efficientnet_b0_ela_p320_s1", "arch": "efficientnet_b0"}])
    out = tmp_path / "out"

    counts = merge([str(a)], str(out), keep_retries=True)

    assert counts["models"] == 1 and counts["log_rows"] == 1
