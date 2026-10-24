"""S0 gate: refuse to start a sweep unless the split and ground-truth masks are the expected ones."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.prepare_dataset import split_fingerprint

# Verified on the real CASIA 2.0 dataset: Kaggle S0 run of 2026-10-24 kept 11,803
# images after removing 811 near-duplicates (sha256 79e78eb2…). A mismatch now means
# the data or the splitter changed — update this table, never re-derive the split to
# fit it.
EXPECTED = {
    "train": (9442, 5253),
    "val": (1180, 923),
    "test": (1181, 878),
}
EXPECTED_IS_PROVISIONAL = False


def _mask_available(raw_dir) -> bool:
    """True if any directory under raw_dir is named *Groundtruth* or ends in _gt."""
    try:
        for p in Path(raw_dir).rglob("*"):
            name = p.name.lower()
            if p.is_dir() and ("groundtruth" in name or name.endswith("_gt")):
                return True
    except OSError:
        return False
    return False


def _read_fingerprint(data_dir):
    """Return (fingerprint, read_problem); a missing CSV becomes a problem, not a traceback."""
    try:
        return split_fingerprint(data_dir), None
    except OSError as e:
        name = Path(getattr(e, "filename", None) or "").name
        return {"counts": {}, "class_counts": {}, "sha256": None}, (
            f"{name or 'split CSVs'} missing or unreadable: {e}")
    except ValueError as e:
        return {"counts": {}, "class_counts": {}, "sha256": None}, (
            f"split CSVs are malformed: {e}")


def verify_setup(data_dir, raw_dir, expected=None) -> dict:
    """Compare the observed split fingerprint against expected (total, authentic) per split."""
    provisional = expected is None and EXPECTED_IS_PROVISIONAL
    expected = EXPECTED if expected is None else expected
    mark = " (provisional)" if provisional else ""

    fp, read_problem = _read_fingerprint(data_dir)
    problems = [read_problem] if read_problem else []

    if not read_problem:
        for split, (n_expected, auth_expected) in expected.items():
            n = fp["counts"].get(split, 0)
            auth = fp["class_counts"].get(split, {}).get("0", 0)
            if n != n_expected or auth != auth_expected:
                problems.append(
                    f"{split}: expected {n_expected} images / {auth_expected} authentic{mark}, "
                    f"observed {n} / {auth}")
        for split, classes in fp["class_counts"].items():
            bad = sorted(set(classes) - {"0", "1"})
            if bad:
                problems.append(
                    f"{split}: unexpected labels {bad}; only 0 (authentic) and "
                    "1 (tampered) are valid")

    return {
        "split_ok": not problems,
        "mask_available": _mask_available(raw_dir),
        "counts": fp["counts"],
        "class_counts": fp["class_counts"],
        "sha256": fp["sha256"],
        "problems": problems,
        "expected": expected,
        "expected_provisional": provisional,
    }


def main():
    p = argparse.ArgumentParser(
        description="S0 gate: verify split counts and ground-truth mask availability")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--raw-dir", default="data/raw")
    args = p.parse_args()

    r = verify_setup(args.data_dir, args.raw_dir)
    print(json.dumps({
        "observed": {
            "counts": r["counts"],
            "class_counts": r["class_counts"],
            "sha256": r["sha256"],
        },
        "expected": {s: {"images": n, "authentic": a}
                     for s, (n, a) in r["expected"].items()},
        "expected_provisional": r["expected_provisional"],
        "split_ok": r["split_ok"],
        "mask_available": r["mask_available"],
        "problems": r["problems"],
    }, indent=2))
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
