"""S0 gate: refuse to start a sweep unless the split and ground-truth masks are the expected ones."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.prepare_dataset import split_fingerprint

# Provisional: Task 1's sorted collect_images changed the split relative to the design
# document, so these counts must be confirmed by the first verified Kaggle run.
PROVISIONAL_EXPECTED = {
    "train": (9448, 5223),
    "val": (1181, 903),
    "test": (1182, 927),
}


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


def verify_setup(data_dir, raw_dir, expected=None) -> dict:
    """Compare the observed split fingerprint against expected (total, authentic) per split."""
    expected = PROVISIONAL_EXPECTED if expected is None else expected
    fp = split_fingerprint(data_dir)

    problems = []
    for split, (n_expected, auth_expected) in expected.items():
        n = fp["counts"].get(split, 0)
        auth = fp["class_counts"].get(split, {}).get("0", 0)
        if n != n_expected or auth != auth_expected:
            problems.append(
                f"{split}: expected {n_expected} images / {auth_expected} authentic, "
                f"observed {n} / {auth}")

    return {
        "split_ok": not problems,
        "mask_available": _mask_available(raw_dir),
        "counts": fp,
        "problems": problems,
    }


def main():
    p = argparse.ArgumentParser(
        description="S0 gate: verify split counts and ground-truth mask availability")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--raw-dir", default="data/raw")
    args = p.parse_args()

    r = verify_setup(args.data_dir, args.raw_dir)
    print(json.dumps({
        "observed": r["counts"],
        "expected": {s: {"images": n, "authentic": a}
                     for s, (n, a) in PROVISIONAL_EXPECTED.items()},
        "split_ok": r["split_ok"],
        "mask_available": r["mask_available"],
        "problems": r["problems"],
    }, indent=2))
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
