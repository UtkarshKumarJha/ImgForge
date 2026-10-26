"""Resumable ablation sweep: 4 archs x 3 input_modes x 3 seeds = 36 runs.

Iterates seed-by-seed so a partial sweep covers every architecture.
Checks experiment_log.csv to skip completed combos.
Persists ELA cache, predictions, checkpoints and experiment_log.csv to
Drive/Kaggle between runs. `--gpus N` trains N combos at once, one per GPU.
"""

import argparse
import contextlib
import csv
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product
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

# The configuration the sweep reports. A run that departs from it gets the
# deviation in its name, so a pilot arm can never masquerade as a reported run.
# Scheduled from the measured T4 cost: EfficientNet-B0/ELA at 256 px takes ~58 s per
# train epoch, so 60 epochs over 36 runs would need ~45 h — more than the 30 h weekly
# GPU quota, which also has to pay for the pilot. 224 px at 30 epochs fits the whole
# grid with slack for reruns, and the 10-epoch pilot tied 224 (0.8104 val macro-F1)
# against 256 (0.8066).
REFERENCE = {
    "image_size": 224,   # the pilot's validation winner (0.8104 vs 0.8066 at 256)
    "fusion": "dual",
    "num_epochs": 60,    # the declared cap; early stopping at patience 10 ends most runs sooner
    "batch_size": 16,
    "lr_new": 1e-3,
    "warmup": 5,
    "grad_clip": 1.0,
    "ela_quality": 75,
}
# Short tag per non-default setting; fusion reads as the bare value (e.g. `concat`).
_TAGS = {"image_size": "p", "num_epochs": "e", "batch_size": "b", "lr_new": "lr",
         "warmup": "w", "grad_clip": "gc", "ela_quality": "q", "fusion": ""}


def deviation_suffix(args) -> str:
    """`_p224` / `_concat` / `_p224_concat` when the config departs from REFERENCE."""
    tags = []
    for key, ref in REFERENCE.items():
        value = getattr(args, key)
        if value == ref:
            continue
        tag = _TAGS[key]
        tags.append(f"{tag}{value}" if tag else str(value))
    if not getattr(args, "tta", True):
        tags.append("notta")
    return "_" + "_".join(tags) if tags else ""


def run_name_for(arch: str, input_mode: str, seed: int, args) -> str:
    """`{arch}_{mode}[_deviations]_s{seed}`.

    The `_s{seed}` suffix must stay last: analysis/mcnemar.py splits run names
    on it to recover the seed.
    """
    return f"{arch}_{input_mode}{deviation_suffix(args)}_s{seed}"


def load_completed(log_path: str) -> set[str]:
    """Run names already logged. Logs predating the run_name column fall back to
    the (arch, input_mode, seed) triple, which only matches undeviated runs."""
    completed = set()
    if not os.path.exists(log_path):
        return completed
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(row.get("run_name")
                          or f"{row['arch']}_{row['input_mode']}_s{row['seed']}")
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
                     run_name: str, predictions_dir: str = "predictions",
                     models_dir: str = "models"):
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
            # Overwrite: a re-run of the same name must not keep the old decisions.
            shutil.copy2(str(npz), os.path.join(dest_preds, npz.name))
            count += 1
        if count:
            print(f"  Persisted {count} prediction files")
        expected = os.path.join(dest_preds, f"{run_name}_test.npz")
        if not os.path.exists(expected):
            print(f"  WARNING: no predictions for {run_name} in {predictions_dir} "
                  f"(missing {run_name}_test.npz) — this run cannot be audited later")
    else:
        print(f"  WARNING: predictions dir not found: {predictions_dir} — "
              f"no prediction artefacts persisted for {run_name}")

    # The checkpoint is what the McNemar analysis selects representatives from,
    # so an unpinned Colab runtime loses the ability to run it at all.
    ckpt = os.path.join(models_dir, f"{run_name}_best.pth")
    dest_models = os.path.join(persist_dir, os.path.basename(os.path.normpath(models_dir)))
    if os.path.exists(ckpt):
        os.makedirs(dest_models, exist_ok=True)
        dest = os.path.join(dest_models, os.path.basename(ckpt))
        shutil.copy2(ckpt, dest)          # overwrite: a re-run supersedes the old one
        print(f"  Persisted {ckpt} -> {dest}")
    else:
        print(f"  WARNING: checkpoint not found: {ckpt} — the McNemar analysis "
              f"cannot select this run's representative without it")

    if os.path.exists(log_path):
        dest = os.path.join(persist_dir, os.path.basename(log_path))
        shutil.copy2(log_path, dest)
        print(f"  Persisted {log_path} -> {dest}")


def restore_from_drive(log_path: str, cache_dir: str | None, persist_dir: str | None,
                       models_dir: str = "models"):
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

    persisted_models = os.path.join(persist_dir, os.path.basename(os.path.normpath(models_dir)))
    if os.path.isdir(persisted_models):
        os.makedirs(models_dir, exist_ok=True)
        restored = 0
        for pth in Path(persisted_models).glob("*.pth"):
            dest = os.path.join(models_dir, pth.name)
            if not os.path.exists(dest):     # a local checkpoint is newer than the copy
                shutil.copy2(str(pth), dest)
                restored += 1
        if restored:
            print(f"Restored {restored} checkpoints from {persisted_models}")


def build_train_cmd(args, arch: str, input_mode: str, seed: int,
                    run_name: str) -> list[str]:
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
        "--run-name", run_name,
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


def run_combo(arch: str, input_mode: str, seed: int, args, gpu: int | None = None,
              lock=None) -> bool:
    """Train, evaluate and log one combo. With `gpu` set the subprocesses are pinned
    to that device and write to results/logs/<run>.log instead of stdout."""
    run_name = run_name_for(arch, input_mode, seed, args)
    ckpt_path = os.path.join(args.out_dir, f"{run_name}_best.pth")
    json_path = os.path.join(args.results_dir, f"{run_name}.json")
    run_log = os.path.join(args.results_dir, "logs", f"{run_name}.log")

    def run(cmd) -> subprocess.CompletedProcess:
        if gpu is None:
            return subprocess.run(cmd)
        os.makedirs(os.path.dirname(run_log), exist_ok=True)
        with open(run_log, "a") as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})

    print(f"\n{'='*60}")
    print(f"TRAIN: {run_name}" + ("" if gpu is None else f" on GPU {gpu} (log: {run_log})"))
    print(f"{'='*60}")
    free = _disk_report("before training")
    if free < 0.5:
        print("SKIPPED: not enough disk space to save checkpoint")
        return False

    train_cmd = build_train_cmd(args, arch, input_mode, seed, run_name)

    ret = run(train_cmd)
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
    ret = run(eval_cmd)
    if ret.returncode != 0:
        print(f"FAILED: evaluate.py returned {ret.returncode}")
        return False

    log_cmd = [
        sys.executable, "experiment_log.py", json_path,
        "--log", args.log_path,
    ]

    print(f"\nLOG: {run_name}")
    with lock or contextlib.nullcontext():  # concurrent runs share one log file
        ret = run(log_cmd)
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
    p.add_argument("--num-epochs", type=int, default=REFERENCE["num_epochs"])
    p.add_argument("--batch-size", type=int, default=REFERENCE["batch_size"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, nargs="+", default=[REFERENCE["image_size"]],
                   help="One or more input sizes; each is a separate (suffixed) run")
    p.add_argument("--lr-new", type=float, default=REFERENCE["lr_new"])
    p.add_argument("--warmup", type=int, default=REFERENCE["warmup"])
    p.add_argument("--grad-clip", type=float, default=REFERENCE["grad_clip"])
    p.add_argument("--fusion", default=REFERENCE["fusion"], choices=["dual", "concat"])
    p.add_argument("--no-tta", action="store_false", dest="tta", default=True,
                   help="Disable test-time augmentation (on by default)")
    p.add_argument("--ela-quality", type=int, default=REFERENCE["ela_quality"])
    p.add_argument("--ela-cache-dir", default="data/ela_cache")
    p.add_argument("--no-persist-ela-cache", action="store_true",
                   help="Keep the ELA cache local to this session. It is ~1.2 MB per "
                        "image, so persisting ~12k images would exceed a Kaggle output.")
    p.add_argument("--persist-dir", default=None,
                   help="Persistent storage (e.g. /content/drive/MyDrive/ImgForge)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--archs", nargs="+", default=ARCHS,
                   choices=ARCHS)
    p.add_argument("--input-modes", nargs="+", default=INPUT_MODES,
                   choices=INPUT_MODES)
    p.add_argument("--max-hours", type=float, default=None,
                   help="Stop starting new runs after this many hours so a capped "
                        "Kaggle session still finishes and saves persist/")
    p.add_argument("--skip-ela-warmup", action="store_true",
                   help="Skip ELA cache warmup (use if cache is already populated)")
    p.add_argument("--array-cache-dir", default=None,
                   help="Build pre-resized RGB/ELA arrays here once and train from them "
                        "(replaces the per-image ELA cache)")
    p.add_argument("--gpus", type=int, default=1,
                   help="Train this many combos concurrently, one per GPU")
    args = p.parse_args()

    if args.num_epochs > REFERENCE["num_epochs"]:
        p.error(f"--num-epochs above {REFERENCE['num_epochs']} is outside the "
                f"pre-declared sweep (a longer schedule is a different experiment)")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.array_cache_dir:
        from data.dataset import ARRAY_CACHE_ENV, build_array_cache
        os.environ[ARRAY_CACHE_ENV] = args.array_cache_dir
    elif args.ela_cache_dir:
        os.environ["IMGFORGE_ELA_CACHE"] = args.ela_cache_dir

    if not args.persist_dir:
        print("WARNING: --persist-dir is unset — results and predictions will NOT "
              "survive this session!")

    restore_from_drive(args.log_path, args.ela_cache_dir, args.persist_dir,
                       args.out_dir)

    needs_ela = any(m in ("ela", "rgb_ela") for m in args.input_modes)
    if args.array_cache_dir:
        for size in args.image_size:
            build_array_cache(args.data_dir, size, args.ela_quality, args.array_cache_dir)
    elif needs_ela and args.ela_cache_dir and not args.skip_ela_warmup:
        warmup_ela_cache(args.data_dir, args.ela_quality, args.ela_cache_dir)

    # One Namespace per job so run names and commands see that job's image size.
    jobs = [(arch, mode, seed, argparse.Namespace(**{**vars(args), "image_size": size}))
            for size, seed, arch, mode in product(
                args.image_size, args.seeds, args.archs, args.input_modes)]
    if args.gpus > 1:
        # Longest runs first, so the last job to finish is a short one.
        cost = {"swin_tiny": 3, "efficientnet_b0": 2, "resnet18_bilstm": 1, "mlp": 0}
        jobs.sort(key=lambda j: -cost[j[0]])

    completed = load_completed(args.log_path)
    total = len(jobs)
    already = sum(1 for a, m, s, j in jobs if run_name_for(a, m, s, j) in completed)
    print(f"\nSweep: {len(args.image_size)} sizes x {len(args.archs)} archs x "
          f"{len(args.input_modes)} modes x {len(args.seeds)} seeds = {total} combos "
          f"on {args.gpus} GPU(s)")
    print(f"Already completed: {already}")
    print(f"Remaining: {total - already}")

    lock = threading.Lock()
    free_gpus = queue.Queue()
    for gpu in range(args.gpus):
        free_gpus.put(gpu)
    started = time.monotonic()

    def work(job) -> bool | None:
        arch, input_mode, seed, job_args = job
        run_name = run_name_for(arch, input_mode, seed, job_args)
        if run_name in completed:
            print(f"\nSKIP: {run_name} (already in log)")
            return None

        elapsed_h = (time.monotonic() - started) / 3600
        if args.max_hours and elapsed_h >= args.max_hours:
            print(f"\nTIME BUDGET: {elapsed_h:.1f} h used >= {args.max_hours} h — "
                  f"not starting {run_name}; finished runs are persisted")
            return None

        gpu = free_gpus.get()
        try:
            ok = run_combo(arch, input_mode, seed, job_args,
                           gpu=gpu if args.gpus > 1 else None, lock=lock)
        finally:
            free_gpus.put(gpu)
        if ok:
            with lock:
                completed.add(run_name)
                persist_to_drive(
                    args.log_path,
                    None if args.no_persist_ela_cache else args.ela_cache_dir,
                    args.persist_dir, run_name, models_dir=args.out_dir)
                run_log = os.path.join(args.results_dir, "logs", f"{run_name}.log")
                if args.persist_dir and os.path.exists(run_log):
                    os.makedirs(os.path.join(args.persist_dir, "logs"), exist_ok=True)
                    shutil.copy2(run_log, os.path.join(args.persist_dir, "logs"))
        print(f"{'DONE' if ok else 'FAILED'}: {run_name}")
        return ok

    with ThreadPoolExecutor(max_workers=args.gpus) as pool:
        results = list(pool.map(work, jobs))
    done, failed = results.count(True), results.count(False)

    print(f"\n{'='*60}")
    print(f"SWEEP DONE: {done} new, {failed} failed, "
          f"{already} previously done, {total} total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
