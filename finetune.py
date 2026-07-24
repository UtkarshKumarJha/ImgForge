import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import numpy as np
import os
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

from data.ela import compute_ela

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224
CHECKPOINT    = "models/best_model.pth"
SAVE_PATH     = "models/docforge_id_finetuned.pth"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {
    "num_epochs":   40,
    "lr":           5e-5,
    "weight_decay": 1e-2,
    "batch_size":   4,
    "patience":     15,
    "run_name":     "docforge-finetune-id-docs",
}


# ── Dataset ───────────────────────────────────────────────────────────────────
class IDDocDataset(Dataset):
    def __init__(self, authentic_dir: str, tampered_dir: str):
        self.samples = []

        for f in os.listdir(authentic_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.samples.append((
                    os.path.join(authentic_dir, f), 0
                ))

        for f in os.listdir(tampered_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.samples.append((
                    os.path.join(tampered_dir, f), 1
                ))

        print(f"ID Dataset: {len(self.samples)} samples "
              f"({sum(1 for _,l in self.samples if l==0)} authentic, "
              f"{sum(1 for _,l in self.samples if l==1)} tampered)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load RGB
        rgb = np.array(
            Image.open(img_path).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
            ), dtype=np.float32
        ) / 255.0

        # Normalize
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        rgb_norm = (rgb - mean) / std
        rgb_t = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()

        # ELA channel
        ela = compute_ela(img_path)
        ela_resized = np.array(
            Image.fromarray((ela * 255).astype(np.uint8)).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
            ), dtype=np.float32
        ) / 255.0
        ela_gray = ela_resized.mean(axis=2)
        ela_gray = (ela_gray - 0.5) / 0.5
        ela_t = torch.from_numpy(ela_gray).unsqueeze(0).float()

        return torch.cat([rgb_t, ela_t], dim=0), torch.tensor(label, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.efficientnet_b0(weights=None)
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        4, old_conv.out_channels,
        old_conv.kernel_size, old_conv.stride,
        old_conv.padding, bias=False
    )
    model.features[0][0] = new_conv
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )
    return model


# ── Load pretrained checkpoint ────────────────────────────────────────────────
model = build_model()
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded checkpoint — base val F1: {ckpt['val_f1']:.4f}")

# Freeze backbone, only train classifier head for first 10 epochs
for param in model.features.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

model.to(DEVICE)


# ── Data ──────────────────────────────────────────────────────────────────────
dataset = IDDocDataset(
    authentic_dir="data/id_synthetic/authentic",
    tampered_dir="data/id_synthetic/tampered"
)

# With only 8 samples, use all for training + manual val on same set
# (acceptable for demo fine-tuning with this dataset size)
loader = DataLoader(
    dataset, batch_size=CFG["batch_size"],
    shuffle=True, num_workers=0
)


# ── Training ──────────────────────────────────────────────────────────────────
wandb.init(project="docforge", name=CFG["run_name"], config=CFG)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG["lr"], weight_decay=CFG["weight_decay"]
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG["num_epochs"]
)

best_loss = float("inf")
patience_ctr = 0
os.makedirs("models", exist_ok=True)

for epoch in range(1, CFG["num_epochs"] + 1):

    # Unfreeze backbone after epoch 10
    if epoch == 11:
        print("Unfreezing backbone for full fine-tuning...")
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CFG["lr"] / 5,
            weight_decay=CFG["weight_decay"]
        )

    model.train()
    epoch_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(loader, desc=f"Epoch {epoch}/{CFG['num_epochs']}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    scheduler.step()

    epoch_loss /= len(dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    wandb.log({
        "epoch": epoch,
        "loss": epoch_loss,
        "acc": acc,
        "f1": f1,
        "lr": scheduler.get_last_lr()[0]
    })

    print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} "
          f"Acc: {acc:.4f} F1: {f1:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_ctr = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": epoch_loss,
            "acc": acc,
            "f1": f1,
            "cfg": CFG
        }, SAVE_PATH)
        print(f"  ✓ Best model saved (loss={epoch_loss:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= CFG["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\nFine-tuning complete. Model saved to {SAVE_PATH}")
wandb.finish()