import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verify_setup import main, verify_setup


def test_mismatched_counts_are_reported_as_problems(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    with open(d / "train.csv", "w") as f:
        f.write("path,label\n/x/1.jpg,0\n")
    for s in ("val", "test"):
        (d / f"{s}.csv").write_text("path,label\n")
    r = verify_setup(str(d), str(tmp_path / "missing_raw"))
    assert r["split_ok"] is False
    assert any("train" in p for p in r["problems"])


def test_expected_counts_pass(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    spec = {"train": (9448, 5223), "val": (1181, 903), "test": (1182, 927)}
    for split, (n, n_auth) in spec.items():
        with open(d / f"{split}.csv", "w") as f:
            f.write("path,label\n")
            for i in range(n):
                f.write(f"/x/{i}.jpg,{0 if i < n_auth else 1}\n")
    r = verify_setup(str(d), str(tmp_path / "missing_raw"))
    assert r["split_ok"] is True
    assert r["mask_available"] is False


def _write_split(d, split, n, n_auth):
    with open(d / f"{split}.csv", "w") as f:
        f.write("path,label\n")
        for i in range(n):
            f.write(f"/x/{i}.jpg,{0 if i < n_auth else 1}\n")


def test_one_problem_per_split_names_expected_and_observed(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    _write_split(d, "train", 3, 2)
    _write_split(d, "val", 0, 0)
    _write_split(d, "test", 0, 0)
    expected = {"train": (7, 4), "val": (0, 0), "test": (0, 0)}
    r = verify_setup(str(d), str(tmp_path / "missing_raw"), expected)
    assert r["split_ok"] is False
    assert len(r["problems"]) == 1
    p = r["problems"][0]
    assert "train" in p
    assert "expected 7 images / 4 authentic" in p
    assert "observed 3 / 2" in p


def test_authentic_only_mismatch_is_a_problem(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    _write_split(d, "train", 5, 5)
    _write_split(d, "val", 0, 0)
    _write_split(d, "test", 0, 0)
    expected = {"train": (5, 3), "val": (0, 0), "test": (0, 0)}
    r = verify_setup(str(d), str(tmp_path / "missing_raw"), expected)
    assert r["split_ok"] is False
    assert len(r["problems"]) == 1
    assert "observed 5 / 5" in r["problems"][0]


@pytest.mark.parametrize("name", ["Groundtruth", "casia_gt", "CASIA2.0_Groundtruth"])
def test_mask_directory_is_detected(tmp_path, name):
    d = tmp_path / "data"; d.mkdir()
    for split in ("train", "val", "test"):
        _write_split(d, split, 0, 0)
    raw = tmp_path / "raw"
    (raw / name).mkdir(parents=True)
    r = verify_setup(str(d), str(raw), {"train": (0, 0), "val": (0, 0), "test": (0, 0)})
    assert r["mask_available"] is True


def test_cli_prints_fingerprint_and_exits_1_on_problems(tmp_path, capsys, monkeypatch):
    d = tmp_path / "data"; d.mkdir()
    _write_split(d, "train", 1, 1)
    for split in ("val", "test"):
        _write_split(d, split, 0, 0)
    monkeypatch.setattr(sys, "argv", ["verify_setup.py", "--data-dir", str(d),
                                      "--raw-dir", str(tmp_path / "missing_raw")])
    assert main() == 1
    out = json.loads(capsys.readouterr().out)
    assert out["observed"]["counts"] == {"train": 1, "val": 0, "test": 0}
    assert out["observed"]["class_counts"]["train"] == {"0": 1}
    assert len(out["observed"]["sha256"]) == 64
    assert out["expected"]["train"] == {"images": 9448, "authentic": 5223}
    assert out["mask_available"] is False
    assert out["problems"]


def test_cli_exits_0_when_split_and_masks_are_expected(tmp_path, capsys, monkeypatch):
    d = tmp_path / "data"; d.mkdir()
    spec = {"train": (9448, 5223), "val": (1181, 903), "test": (1182, 927)}
    for split, (n, n_auth) in spec.items():
        _write_split(d, split, n, n_auth)
    (tmp_path / "raw" / "Groundtruth").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["verify_setup.py", "--data-dir", str(d),
                                      "--raw-dir", str(tmp_path / "raw")])
    assert main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["problems"] == []
    assert out["mask_available"] is True


def test_documented_script_invocation_exits_1_on_a_bad_split(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    _write_split(d, "train", 1, 1)
    for split in ("val", "test"):
        _write_split(d, split, 0, 0)
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_setup.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(d),
         "--raw-dir", str(tmp_path / "missing_raw")],
        capture_output=True, text=True)
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["problems"]
