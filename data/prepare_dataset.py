import random
import csv
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data")
SPLIT = (0.80, 0.10, 0.10)
SEED = 42

random.seed(SEED)

def collect_images(folder, label):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
    return [(str(p), label) for p in Path(folder).rglob("*") if p.suffix.lower() in exts]

authentic = collect_images(RAW_DIR / "CASIA2.0_revised" / "Au", label=0)
tampered  = collect_images(RAW_DIR / "CASIA2.0_revised" / "Tp", label=1)
print(f"Authentic: {len(authentic)}  |  Tampered: {len(tampered)}")

all_samples = authentic + tampered
random.shuffle(all_samples)

n = len(all_samples)
n_train = int(n * SPLIT[0])
n_val   = int(n * SPLIT[1])

splits = {
    "train": all_samples[:n_train],
    "val":   all_samples[n_train:n_train + n_val],
    "test":  all_samples[n_train + n_val:]
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

for split, samples in splits.items():
    csv_path = OUT_DIR / f"{split}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(samples)
    auth = sum(1 for _, l in samples if l == 0)
    tamp = sum(1 for _, l in samples if l == 1)
    print(f"{split:5s}: {len(samples)} total | authentic={auth} tampered={tamp} → {csv_path}")

print("Done — CSV index files created, no files copied.")