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
    spec = {"train": (9442, 5253), "val": (1180, 923), "test": (1181, 878)}
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


def test_existing_raw_dir_without_masks_is_false(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    for split in ("train", "val", "test"):
        _write_split(d, split, 0, 0)
    raw = tmp_path / "raw"
    (raw / "CASIA2.0_revised" / "Au").mkdir(parents=True)
    r = verify_setup(str(d), str(raw), {"train": (0, 0), "val": (0, 0), "test": (0, 0)})
    assert r["mask_available"] is False


def test_missing_split_csv_is_a_problem_not_a_traceback(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    _write_split(d, "train", 1, 1)
    r = verify_setup(str(d), str(tmp_path / "missing_raw"), {"train": (1, 1)})
    assert r["split_ok"] is False
    assert any("val.csv missing or unreadable" in p for p in r["problems"])
    assert r["sha256"] is None


def test_missing_label_column_is_a_problem_not_a_traceback(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    (d / "train.csv").write_text("path\n/x/0.jpg\n")
    for split in ("val", "test"):
        _write_split(d, split, 0, 0)

    r = verify_setup(str(d), str(tmp_path / "missing_raw"), {"train": (1, 1)})

    assert r["split_ok"] is False
    assert any("train.csv lacks column(s): label" in p for p in r["problems"])
    assert r["sha256"] is None


def test_unexpected_label_is_a_problem(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    (d / "train.csv").write_text("path,label\n/x/0.jpg,0\n/x/1.jpg,0\n/x/2.jpg,2\n")
    for split in ("val", "test"):
        _write_split(d, split, 0, 0)
    r = verify_setup(str(d), str(tmp_path / "missing_raw"),
                     {"train": (3, 2), "val": (0, 0), "test": (0, 0)})
    assert r["split_ok"] is False
    assert len(r["problems"]) == 1
    assert "train" in r["problems"][0]
    assert "unexpected labels ['2']" in r["problems"][0]


def test_cli_uses_the_verified_table_and_exits_1_on_problems(tmp_path, capsys, monkeypatch):
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
    assert out["expected"]["train"] == {"images": 9442, "authentic": 5253}
    assert out["expected_provisional"] is False
    assert out["mask_available"] is False
    assert all("(provisional)" not in p for p in out["problems"])


def test_custom_expected_is_echoed_without_provisional_marker(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    _write_split(d, "train", 4, 3)
    _write_split(d, "val", 0, 0)
    _write_split(d, "test", 0, 0)
    expected = {"train": (9, 3), "val": (0, 0), "test": (0, 0)}
    r = verify_setup(str(d), str(tmp_path / "missing_raw"), expected)
    assert r["expected"] == expected
    assert r["expected_provisional"] is False
    assert "provisional" not in r["problems"][0]
    assert r["counts"] == {"train": 4, "val": 0, "test": 0}
    assert r["class_counts"]["train"] == {"0": 3, "1": 1}
    assert len(r["sha256"]) == 64


def test_cli_exits_0_when_split_and_masks_are_expected(tmp_path, capsys, monkeypatch):
    d = tmp_path / "data"; d.mkdir()
    spec = {"train": (9442, 5253), "val": (1180, 923), "test": (1181, 878)}
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
