"""Prepare CASIA 2.0 dataset: dedup, source-group, split into train/val/test CSVs."""

import argparse
import csv
import re
import random
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from data.ela import read_jpeg_quality

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data")
SPLIT = (0.80, 0.10, 0.10)
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
HASH_SIZE = 16
HAMMING_THRESHOLD = 6


def collect_images(folder, label):
    return [(str(p), label) for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS]


def compute_phash(path):
    try:
        return imagehash.phash(Image.open(path).convert("RGB"), hash_size=HASH_SIZE)
    except Exception as e:
        print(f"  WARN: cannot hash {path}: {e}")
        return None


def dedup_by_phash(samples):
    """Remove near-duplicate images (Hamming distance ≤ threshold). Keep first seen."""
    seen_hashes = []
    kept, removed = [], 0
    for path, label in samples:
        h = compute_phash(path)
        if h is None:
            kept.append((path, label))
            continue
        is_dup = any(h - existing <= HAMMING_THRESHOLD for existing in seen_hashes)
        if is_dup:
            removed += 1
        else:
            seen_hashes.append(h)
            kept.append((path, label))
    print(f"  Dedup: removed {removed} near-duplicates, kept {len(kept)}")
    return kept


_TP_RE = re.compile(
    r"Tp_[A-Z]_[A-Z]{3}_[A-Z]_[A-Z]_([a-z]+\d+)_([a-z]+\d+)_\d+"
)


def extract_tp_sources(filename):
    """Extract (donor, host) source IDs from a CASIA Tp filename."""
    m = _TP_RE.match(Path(filename).stem)
    if m:
        return m.group(1), m.group(2)
    return None


def group_split(samples, rng):
    """Split samples at the source-group level so related images stay together.

    Tp images sharing a donor or host source are grouped via union-find.
    Au images each get their own singleton group.
    """
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    source_to_indices = defaultdict(list)
    for i, (path, label) in enumerate(samples):
        parent[i] = i
        srcs = extract_tp_sources(path)
        if srcs:
            for s in srcs:
                source_to_indices[s].append(i)

    for indices in source_to_indices.values():
        for j in range(1, len(indices)):
            union(indices[0], indices[j])

    groups = defaultdict(list)
    for i in range(len(samples)):
        groups[find(i)].append(samples[i])

    group_list = list(groups.values())
    rng.shuffle(group_list)

    n_total = sum(len(g) for g in group_list)
    n_train_target = int(n_total * SPLIT[0])
    n_val_target = int(n_total * SPLIT[1])

    splits = {"train": [], "val": [], "test": []}
    running = 0
    for g in group_list:
        if running < n_train_target:
            splits["train"].extend(g)
        elif running < n_train_target + n_val_target:
            splits["val"].extend(g)
        else:
            splits["test"].extend(g)
        running += len(g)

    return splits


def _log_qf_distribution(authentic, tampered):
    """Print JPEG quality-factor stats per class to flag ELA confounds."""
    import statistics
    for class_name, samples in [("Authentic", authentic), ("Tampered", tampered)]:
        qfs = []
        non_jpeg = 0
        for path, _ in samples:
            qf = read_jpeg_quality(path)
            if qf is not None:
                qfs.append(qf)
            else:
                non_jpeg += 1
        if qfs:
            print(f"\n{class_name} (n={len(qfs)} JPEG, {non_jpeg} non-JPEG):")
            print(f"  mean QF = {statistics.mean(qfs):.1f}")
            print(f"  median  = {statistics.median(qfs):.1f}")
            print(f"  stdev   = {statistics.stdev(qfs):.1f}" if len(qfs) > 1 else "")
            print(f"  min/max = {min(qfs)}/{max(qfs)}")
            # Histogram buckets
            buckets = [0]*10
            for q in qfs:
                buckets[min(q // 10, 9)] += 1
            for b in range(10):
                lo, hi = b*10, (b+1)*10 - 1
                bar = "#" * buckets[b]
                print(f"  [{lo:3d}-{hi:3d}]: {buckets[b]:4d} {bar}")
        else:
            print(f"\n{class_name}: no JPEG files ({non_jpeg} non-JPEG)")
    print("\nIf Au/Tp QF distributions differ strongly, ELA at a fixed"
          " recompression quality will encode class identity, not forgery."
          "\nConsider setting --ela-quality far from both medians, or"
          " use per-image quality matching.")


def main():
    parser = argparse.ArgumentParser(description="Prepare CASIA 2.0 dataset splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--log-qf", action="store_true",
                        help="Log JPEG quality factor distribution by class and exit")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    raw = Path(args.raw_dir)
    out = Path(args.out_dir)

    authentic = collect_images(raw / "CASIA2.0_revised" / "Au", label=0)
    tampered = collect_images(raw / "CASIA2.0_revised" / "Tp", label=1)
    print(f"Authentic: {len(authentic)}  |  Tampered: {len(tampered)}")

    if args.log_qf:
        _log_qf_distribution(authentic, tampered)
        return

    all_samples = authentic + tampered
    rng.shuffle(all_samples)

    print("Deduplicating by perceptual hash...")
    all_samples = dedup_by_phash(all_samples)

    print("Splitting by source group...")
    splits = group_split(all_samples, rng)

    tp_sources_by_split = defaultdict(set)
    for split_name, samples in splits.items():
        for path, _ in samples:
            srcs = extract_tp_sources(path)
            if srcs:
                tp_sources_by_split[split_name].update(srcs)

    leaked = set()
    for a in tp_sources_by_split:
        for b in tp_sources_by_split:
            if a < b:
                overlap = tp_sources_by_split[a] & tp_sources_by_split[b]
                if overlap:
                    leaked.update(overlap)
    if leaked:
        print(f"  WARNING: {len(leaked)} source(s) leaked across splits: {leaked}")
    else:
        print("  OK: no shared Tp sources across splits")

    out.mkdir(parents=True, exist_ok=True)
    for split_name, samples in splits.items():
        csv_path = out / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            writer.writerows(samples)
        auth = sum(1 for _, l in samples if l == 0)
        tamp = sum(1 for _, l in samples if l == 1)
        print(f"{split_name:5s}: {len(samples)} total | authentic={auth} tampered={tamp} → {csv_path}")

    print("Done — CSV index files created, no files copied.")


if __name__ == "__main__":
    main()
