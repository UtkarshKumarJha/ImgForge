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


def _disk_free_gb(path: str = ".") -> float:
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / (1024 ** 3)


def _disk_report(label: str = ""):
    free = _disk_free_gb(".")
    tag = f" [{label}]" if label else ""
    print(f"  Disk free{tag}: {free:.1f} GB")
    if free < 1.0:
        print("  WARNING: less than 1 GB free — risk of torch.save failures!")
    return free

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
    _disk_report("before warmup")

    for path in tqdm(sorted(all_paths), desc="Warming ELA cache"):
        compute_ela(path, quality=ela_quality)

    final = sum(1 for _ in Path(cache_dir).glob("*.npy"))
    print(f"ELA cache ready: {final} files")
    _disk_report("after warmup")


def persist_to_drive(log_path: str, cache_dir: str | None, persist_dir: str | None,
                     run_name: str, predictions_dir: str = "predictions"):
    if not persist_dir:
        return
    os.makedirs(persist_dir, exist_ok=True)

    # Artefacts before the log: the log copy is the completion marker, so nothing
    # may be persisted after it or a failed copy would be skipped forever on resume.
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

    if os.path.isdir(predictions_dir):
        dest_preds = os.path.join(
            persist_dir, os.path.basename(os.path.normpath(predictions_dir)))
        os.makedirs(dest_preds, exist_ok=True)
        count = 0
        for npz in Path(predictions_dir).glob("*.npz"):
            dest = os.path.join(dest_preds, npz.name)
            if not os.path.exists(dest):
                shutil.copy2(str(npz), dest)
                count += 1
        if count:
            print(f"  Persisted {count} new prediction files")
        expected = os.path.join(dest_preds, f"{run_name}_test.npz")
        if not os.path.exists(expected):
            print(f"  WARNING: no predictions for {run_name} in {predictions_dir} "
                  f"(missing {run_name}_test.npz) — this run cannot be audited later")
    else:
        print(f"  WARNING: predictions dir not found: {predictions_dir} — "
              f"no prediction artefacts persisted for {run_name}")

    if os.path.exists(log_path):
        dest = os.path.join(persist_dir, os.path.basename(log_path))
        shutil.copy2(log_path, dest)
        print(f"  Persisted {log_path} -> {dest}")


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


def build_train_cmd(args, arch: str, input_mode: str, seed: int) -> list[str]:
    cmd = [
        sys.executable, "train.py",
        "--arch", arch,
        "--input-mode", input_mode,
        "--seed", str(seed),
        "--data-dir", args.data_dir,
        "--image-size", str(args.image_size),
        "--num-epochs", str(args.num_epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--ela-quality", str(args.ela_quality),
        "--lr-new", str(args.lr_new),
        "--warmup", str(args.warmup),
        "--grad-clip", str(args.grad_clip),
        "--fusion", args.fusion,
        "--out-dir", args.out_dir,
    ]
    if args.wandb:
        cmd.append("--wandb")
    return cmd


def build_eval_cmd(args, ckpt_path: str, json_path: str) -> list[str]:
    cmd = [
        sys.executable, "evaluate.py", ckpt_path,
        "--data-dir", args.data_dir,
        "--split", "test",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--ela-quality", str(args.ela_quality),
        "--out-json", json_path,
    ]
    if args.tta:
        cmd.append("--tta")
    return cmd


def run_combo(arch: str, input_mode: str, seed: int, args) -> bool:
    run_name = f"{arch}_{input_mode}_s{seed}"
    ckpt_path = os.path.join(args.out_dir, f"{run_name}_best.pth")
    json_path = os.path.join(args.results_dir, f"{run_name}.json")

    print(f"\n{'='*60}")
    print(f"TRAIN: {run_name}")
    print(f"{'='*60}")
    free = _disk_report("before training")
    if free < 0.5:
        print("SKIPPED: not enough disk space to save checkpoint")
        return False

    train_cmd = build_train_cmd(args, arch, input_mode, seed)

    ret = subprocess.run(train_cmd)
    if ret.returncode != 0:
        print(f"FAILED: train.py returned {ret.returncode}")
        corrupt = ckpt_path
        if os.path.exists(corrupt):
            os.remove(corrupt)
            print(f"  Cleaned up corrupt checkpoint: {corrupt}")
        return False

    if not os.path.exists(ckpt_path):
        print(f"FAILED: checkpoint not found at {ckpt_path}")
        return False

    eval_cmd = build_eval_cmd(args, ckpt_path, json_path)

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

    if os.path.exists(json_path):
        os.remove(json_path)

    _disk_report("after run")
    return True


def main():
    p = argparse.ArgumentParser(description="Resumable ablation sweep")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--log-path", default="results/experiment_log.csv")
    p.add_argument("--num-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--lr-new", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--fusion", default="dual", choices=["dual", "concat"])
    p.add_argument("--no-tta", action="store_false", dest="tta", default=True,
                   help="Disable test-time augmentation (on by default)")
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

    if not args.persist_dir:
        print("WARNING: --persist-dir is unset — results and predictions will NOT "
              "survive this session!")

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
                    persist_to_drive(args.log_path, args.ela_cache_dir,
                                     args.persist_dir,
                                     f"{arch}_{input_mode}_s{seed}")
                else:
                    failed += 1

    print(f"\n{'='*60}")
    print(f"SWEEP DONE: {done} new, {failed} failed, "
          f"{len(completed) - done} previously done, {total} total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
