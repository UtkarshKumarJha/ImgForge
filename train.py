"""Train a forgery-detection model. Config-driven via CLI args."""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from data.dataset import ForgeryDataset
from model_factory import build_model, VALID_ARCHS, VALID_MODES


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="Train forgery detector")
    p.add_argument("--arch", type=str, default="efficientnet_b0", choices=sorted(VALID_ARCHS))
    p.add_argument("--input-mode", type=str, default="rgb_ela", choices=sorted(VALID_MODES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--ela-quality", type=int, default=75)
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--out-dir", type=str, default="models")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"{args.arch}_{args.input_mode}_s{args.seed}"

    cfg = vars(args)
    cfg["device"] = device
    cfg["run_name"] = run_name

    if args.wandb:
        import wandb
        wandb.init(project="imgforge", name=run_name, config=cfg)

    train_ds = ForgeryDataset(
        args.data_dir, split="train", image_size=args.image_size,
        input_mode=args.input_mode, ela_quality=args.ela_quality,
    )
    val_ds = ForgeryDataset(
        args.data_dir, split="val", image_size=args.image_size,
        input_mode=args.input_mode, ela_quality=args.ela_quality,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model(args.arch, args.input_mode).to(device)

    n_auth = sum(1 for _, l in train_ds.samples if l == 0)
    n_tamp = sum(1 for _, l in train_ds.samples if l == 1)
    total = n_auth + n_tamp
    weights = torch.tensor([total / (2 * n_auth), total / (2 * n_tamp)]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best_ckpt = os.path.join(args.out_dir, f"{run_name}_best.pth")
    best_val_f1 = 0.0
    patience_ctr = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_ds)
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        train_f1 = f1_score(train_labels, train_preds, average="macro")

        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_ds)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        scheduler.step()

        log = {
            "epoch": epoch,
            "train/loss": train_loss, "train/acc": train_acc, "train/f1": train_f1,
            "val/loss": val_loss, "val/acc": val_acc, "val/f1": val_f1,
            "lr": scheduler.get_last_lr()[0],
        }

        if args.wandb:
            import wandb
            wandb.log(log)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "arch": args.arch,
                "input_mode": args.input_mode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "cfg": cfg,
            }, best_ckpt)
            print(f"  > New best model saved (val_f1={val_f1:.4f})")
            if args.wandb:
                import wandb
                wandb.save(best_ckpt)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    print(f"Checkpoint: {best_ckpt}")

    if args.wandb:
        import wandb
        wandb.finish()

    return best_ckpt, best_val_f1


if __name__ == "__main__":
    main()
