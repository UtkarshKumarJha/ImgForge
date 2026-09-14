"""Resumable ablation sweep: 4 archs x 3 input_modes x 3 seeds = 36 runs.

Iterates seed-by-seed so a partial sweep covers every architecture.
Checks experiment_log.csv to skip completed combos.
Persists ELA cache + experiment_log.csv to Drive/Kaggle between runs.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

ARCHS = ["efficientnet_b0", "mlp", "resnet18_bilstm", "swin_tiny"]
INPUT_MODES = ["rgb", "ela", "rgb_ela"]
SEEDS = [1, 2, 3]


def load_completed(log_path: str) -> set[tuple[str, str, int]]:
    completed = set()
    if not os.path.exists(log_path):
        return completed
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["arch"], row["input_mode"], int(row["seed"])))
    return completed


def warmup_ela_cache(data_dir: str, ela_quality: int, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    from data.ela import compute_ela

    all_paths = set()
    for split in ("train", "val", "test"):
        csv_path = os.path.join(data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_paths.add(row["path"])

    cached = sum(1 for _ in Path(cache_dir).glob("*.npy"))
    print(f"ELA cache: {len(all_paths)} images, {cached} already cached")

    for path in tqdm(sorted(all_paths), desc="Warming ELA cache"):
        compute_ela(path, quality=ela_quality)

    final = sum(1 for _ in Path(cache_dir).glob("*.npy"))
    print(f"ELA cache ready: {final} files")


def persist_to_drive(log_path: str, cache_dir: str | None, persist_dir: str | None):
    if not persist_dir:
        return
    os.makedirs(persist_dir, exist_ok=True)

    if os.path.exists(log_path):
        dest = os.path.join(persist_dir, os.path.basename(log_path))
        shutil.copy2(log_path, dest)
        print(f"  Persisted {log_path} -> {dest}")

    if cache_dir and os.path.isdir(cache_dir):
        dest_cache = os.path.join(persist_dir, "ela_cache")
        os.makedirs(dest_cache, exist_ok=True)
        count = 0
        for npy in Path(cache_dir).glob("*.npy"):
            dest = os.path.join(dest_cache, npy.name)
            if not os.path.exists(dest):
                shutil.copy2(str(npy), dest)
                count += 1
        if count:
            print(f"  Persisted {count} new ELA cache files")


def restore_from_drive(log_path: str, cache_dir: str | None, persist_dir: str | None):
    if not persist_dir:
        return

    persisted_log = os.path.join(persist_dir, os.path.basename(log_path))
    if os.path.exists(persisted_log) and not os.path.exists(log_path):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        shutil.copy2(persisted_log, log_path)
        print(f"Restored {log_path} from {persisted_log}")

    if cache_dir:
        persisted_cache = os.path.join(persist_dir, "ela_cache")
        if os.path.isdir(persisted_cache):
            os.makedirs(cache_dir, exist_ok=True)
            restored = 0
            for npy in Path(persisted_cache).glob("*.npy"):
                dest = os.path.join(cache_dir, npy.name)
                if not os.path.exists(dest):
                    shutil.copy2(str(npy), dest)
                    restored += 1
            if restored:
                print(f"Restored {restored} ELA cache files from {persisted_cache}")


def run_combo(arch: str, input_mode: str, seed: int, args) -> bool:
    run_name = f"{arch}_{input_mode}_s{seed}"
    ckpt_path = os.path.join(args.out_dir, f"{run_name}_best.pth")
    json_path = os.path.join(args.results_dir, f"{run_name}.json")

    train_cmd = [
        sys.executable, "train.py",
        "--arch", arch,
        "--input-mode", input_mode,
        "--seed", str(seed),
        "--data-dir", args.data_dir,
        "--num-epochs", str(args.num_epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--ela-quality", str(args.ela_quality),
        "--out-dir", args.out_dir,
    ]
    if args.wandb:
        train_cmd.append("--wandb")

    print(f"\n{'='*60}")
    print(f"TRAIN: {run_name}")
    print(f"{'='*60}")
    ret = subprocess.run(train_cmd)
    if ret.returncode != 0:
        print(f"FAILED: train.py returned {ret.returncode}")
        return False

    if not os.path.exists(ckpt_path):
        print(f"FAILED: checkpoint not found at {ckpt_path}")
        return False

    eval_cmd = [
        sys.executable, "evaluate.py", ckpt_path,
        "--data-dir", args.data_dir,
        "--split", "test",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--ela-quality", str(args.ela_quality),
        "--out-json", json_path,
    ]

    print(f"\nEVALUATE: {run_name}")
    ret = subprocess.run(eval_cmd)
    if ret.returncode != 0:
        print(f"FAILED: evaluate.py returned {ret.returncode}")
        return False

    log_cmd = [
        sys.executable, "experiment_log.py", json_path,
        "--log", args.log_path,
    ]

    print(f"\nLOG: {run_name}")
    ret = subprocess.run(log_cmd)
    if ret.returncode != 0:
        print(f"FAILED: experiment_log.py returned {ret.returncode}")
        return False

    return True


def main():
    p = argparse.ArgumentParser(description="Resumable ablation sweep")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--log-path", default="results/experiment_log.csv")
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ela-quality", type=int, default=75)
    p.add_argument("--ela-cache-dir", default="data/ela_cache")
    p.add_argument("--persist-dir", default=None,
                   help="Persistent storage (e.g. /content/drive/MyDrive/ImgForge)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--archs", nargs="+", default=ARCHS,
                   choices=ARCHS)
    p.add_argument("--input-modes", nargs="+", default=INPUT_MODES,
                   choices=INPUT_MODES)
    p.add_argument("--skip-ela-warmup", action="store_true",
                   help="Skip ELA cache warmup (use if cache is already populated)")
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.ela_cache_dir:
        os.environ["IMGFORGE_ELA_CACHE"] = args.ela_cache_dir

    restore_from_drive(args.log_path, args.ela_cache_dir, args.persist_dir)

    needs_ela = any(m in ("ela", "rgb_ela") for m in args.input_modes)
    if needs_ela and args.ela_cache_dir and not args.skip_ela_warmup:
        warmup_ela_cache(args.data_dir, args.ela_quality, args.ela_cache_dir)

    completed = load_completed(args.log_path)
    total = len(args.seeds) * len(args.archs) * len(args.input_modes)
    print(f"\nSweep: {len(args.archs)} archs x {len(args.input_modes)} modes x {len(args.seeds)} seeds = {total} combos")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {total - len(completed)}")

    done, failed = 0, 0
    for seed in args.seeds:
        for arch in args.archs:
            for input_mode in args.input_modes:
                if (arch, input_mode, seed) in completed:
                    print(f"\nSKIP: {arch}_{input_mode}_s{seed} (already in log)")
                    continue

                ok = run_combo(arch, input_mode, seed, args)
                if ok:
                    done += 1
                    completed.add((arch, input_mode, seed))
                    persist_to_drive(args.log_path, args.ela_cache_dir, args.persist_dir)
                else:
                    failed += 1

    print(f"\n{'='*60}")
    print(f"SWEEP DONE: {done} new, {failed} failed, "
          f"{len(completed) - done} previously done, {total} total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
