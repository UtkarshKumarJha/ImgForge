"""Assemble one analysis directory from several sweep accounts' `persist/` downloads.

Each `--src` may be a `persist/` directory or its parent. Files are matched by name, so a
run present in two sources is copied once; the logs are concatenated and de-duplicated on
`run_name`. Retry/ablation runs (`_p320`, `_concat`, ...) are left out of the canonical
directory on purpose: they are reported separately and must not enter the McNemar grid.

Usage:
    python scripts/merge_artefacts.py --src /tmp/opencode/wA/persist \
        --src /tmp/opencode/wB1/persist --src /tmp/opencode/wB2/persist --out /tmp/opencode/merged
"""

import argparse
import csv
import os
import shutil
import sys

# Deviation tags from run_sweep.deviation_suffix that are retries or ablations, not grid runs.
NON_GRID_TAGS = ("_p320", "_concat", "_e10", "_p224")


def _persist_dir(path: str) -> str:
    """Accept either a persist/ directory or the directory that contains one."""
    if os.path.isdir(os.path.join(path, "predictions")) or \
            os.path.isfile(os.path.join(path, "experiment_log.csv")):
        return path
    nested = os.path.join(path, "persist")
    return nested if os.path.isdir(nested) else path


def _is_grid_run(name: str) -> bool:
    return not any(tag in name for tag in NON_GRID_TAGS)


def merge(sources, out_dir: str, keep_retries: bool = False) -> dict:
    """Copy artefacts and merge the logs; returns per-directory file counts."""
    for sub in ("models", "predictions"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    copied = {"models": 0, "predictions": 0}
    rows, seen = [], set()
    for src in sources:
        src = _persist_dir(src)
        # A typo'd or not-yet-downloaded source used to merge silently as zero runs, which is
        # how an incomplete grid can reach McNemar looking complete.
        if not os.path.isdir(os.path.join(src, "predictions")) and \
                not os.path.isfile(os.path.join(src, "experiment_log.csv")):
            raise FileNotFoundError(f"no persist/ artefacts under {src!r}")
        for sub in ("models", "predictions"):
            for name in sorted(os.listdir(os.path.join(src, sub))
                               if os.path.isdir(os.path.join(src, sub)) else []):
                if not keep_retries and not _is_grid_run(name):
                    continue
                dst = os.path.join(out_dir, sub, name)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(src, sub, name), dst)
                    copied[sub] += 1

        log = os.path.join(src, "experiment_log.csv")
        if os.path.isfile(log):
            with open(log, newline="") as f:
                for row in csv.DictReader(f):
                    if keep_retries or _is_grid_run(row["run_name"]):
                        if row["run_name"] not in seen:
                            seen.add(row["run_name"])
                            rows.append(row)

    if rows:
        dst = os.path.join(out_dir, "experiment_log.csv")
        with open(dst, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r["run_name"]))
    copied["log_rows"] = len(rows)
    return copied


def main():
    p = argparse.ArgumentParser(description="Merge sweep persist/ directories for analysis")
    p.add_argument("--src", action="append", required=True, help="persist/ dir or its parent")
    p.add_argument("--out", required=True)
    p.add_argument("--keep-retries", action="store_true",
                   help="Also copy retry/ablation runs (never for the McNemar grid)")
    args = p.parse_args()

    counts = merge(args.src, args.out, args.keep_retries)
    print(f"merged into {args.out}: {counts['models']} checkpoints, "
          f"{counts['predictions']} prediction files, {counts['log_rows']} log rows")


if __name__ == "__main__":
    sys.exit(main())
