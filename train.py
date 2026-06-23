import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

from data.dataset import ForgeryDataset

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "data_dir": "data",
    "image_size":  224,
    "batch_size":  16,
    "num_epochs":  50,
    "lr":          3e-4,
    "weight_decay":1e-2,
    "patience":    10,       # early stopping on val F1
    "num_workers": 0,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
    "run_name":    "docforge-phase2-casia",
}

# ── W&B Init ──────────────────────────────────────────────────────────────────
wandb.init(project="docforge", name=CFG["run_name"], config=CFG)

# ── Dataset & Loaders ─────────────────────────────────────────────────────────
train_ds = ForgeryDataset(CFG["data_dir"], split="train", image_size=CFG["image_size"])
val_ds   = ForgeryDataset(CFG["data_dir"], split="val",   image_size=CFG["image_size"])

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)

# ── Model: EfficientNet-B0 with 4-channel input ───────────────────────────────
def build_model(num_classes: int = 2) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Expand first conv layer from 3 → 4 channels
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    # Copy pretrained RGB weights; initialize ELA channel as mean of RGB weights
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

    model.features[0][0] = new_conv

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model

model = build_model().to(CFG["device"])

# ── Class weights for imbalanced dataset ─────────────────────────────────────
n_auth = sum(1 for _, l in train_ds.samples if l == 0)
n_tamp = sum(1 for _, l in train_ds.samples if l == 1)
total  = n_auth + n_tamp
weights = torch.tensor([total / (2 * n_auth), total / (2 * n_tamp)]).to(CFG["device"])

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["num_epochs"])

# ── Training Loop ─────────────────────────────────────────────────────────────
best_val_f1   = 0.0
patience_ctr  = 0
best_ckpt     = "models/best_model.pth"

import os; os.makedirs("models", exist_ok=True)

for epoch in range(1, CFG["num_epochs"] + 1):

    # ── Train ──
    model.train()
    train_loss, train_preds, train_labels = 0.0, [], []

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['num_epochs']} [Train]"):
        inputs, labels = inputs.to(CFG["device"]), labels.to(CFG["device"])

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_preds.extend(outputs.argmax(1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_ds)
    train_acc   = np.mean(np.array(train_preds) == np.array(train_labels))
    train_f1    = f1_score(train_labels, train_preds, average="macro")

    # ── Validate ──
    model.eval()
    val_loss, val_preds, val_labels = 0.0, [], []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{CFG['num_epochs']} [Val]"):
            inputs, labels = inputs.to(CFG["device"]), labels.to(CFG["device"])
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_preds.extend(outputs.argmax(1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_ds)
    val_acc   = np.mean(np.array(val_preds) == np.array(val_labels))
    val_f1    = f1_score(val_labels, val_preds, average="macro")

    scheduler.step()

    # ── Log to W&B ──
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss, "train/acc": train_acc, "train/f1": train_f1,
        "val/loss":   val_loss,   "val/acc":   val_acc,   "val/f1":   val_f1,
        "lr": scheduler.get_last_lr()[0]
    })

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} "
          f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

    # ── Checkpoint & Early Stopping ──
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_ctr = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_f1,
            "val_acc": val_acc,
            "cfg": CFG
        }, best_ckpt)
        print(f"  ✓ New best model saved (val_f1={val_f1:.4f})")
        wandb.save(best_ckpt)
    else:
        patience_ctr += 1
        if patience_ctr >= CFG["patience"]:
            print(f"Early stopping at epoch {epoch} (no improvement for {CFG['patience']} epochs)")
            break

print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
wandb.finish()