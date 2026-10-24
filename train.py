"""Train a forgery-detection model. Config-driven via CLI args."""

import argparse
import csv
import math
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


# Name prefixes of the pretrained trunks. Everything else is randomly initialised.
BACKBONE_PREFIXES = {
    "efficientnet_b0": ("features.",),
    "swin_tiny": ("features.", "norm."),  # Swin ends in a top-level LayerNorm
    "resnet18_bilstm": ("backbone.",),
    "mlp": (),  # no pretrained trunk
}


def build_param_groups(model: nn.Module, arch: str, lr_backbone: float,
                       lr_new: float) -> list[dict]:
    """Two LR groups: pretrained trunk vs randomly-initialised modules."""
    prefixes = BACKBONE_PREFIXES[arch]
    backbone, new = [], []
    for name, param in model.named_parameters():
        # The dual ELA stem lives under the trunk's prefix but is not pretrained.
        is_backbone = "aux_conv" not in name and name.startswith(prefixes)
        (backbone if is_backbone else new).append(param)

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone})
    groups.append({"params": new, "lr": lr_new})
    return groups


def make_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup then cosine decay to zero, as a single LambdaLR."""
    warmup = max(1, warmup_epochs)

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = min(1.0, (epoch - warmup) / max(1, total_epochs - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def parse_args():
    p = argparse.ArgumentParser(description="Train forgery detector")
    p.add_argument("--arch", type=str, default="efficientnet_b0", choices=sorted(VALID_ARCHS))
    p.add_argument("--input-mode", type=str, default="rgb_ela", choices=sorted(VALID_MODES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--lr-backbone", type=float, default=None,
                   help="LR for pretrained trunk weights (default: 2e-5 for swin_tiny, else 1e-4)")
    p.add_argument("--lr-new", type=float, default=1e-3,
                   help="LR for randomly-initialised modules")
    p.add_argument("--warmup", type=int, default=5, help="Linear warmup epochs")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm (0 disables)")
    p.add_argument("--fusion", type=str, default="dual", choices=["dual", "concat"],
                   help="How the ELA channel joins the RGB input")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--ela-quality", type=int, default=75)
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--out-dir", type=str, default="models")
    p.add_argument("--run-name", type=str, default="",
                   help="Checkpoint name; the sweep passes its deviation-suffixed name")
    args = p.parse_args()
    if args.lr_backbone is None:
        args.lr_backbone = 2e-5 if args.arch == "swin_tiny" else 1e-4
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or f"{args.arch}_{args.input_mode}_s{args.seed}"

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

    model = build_model(args.arch, args.input_mode, fusion=args.fusion).to(device)

    n_auth = sum(1 for _, l in train_ds.samples if l == 0)
    n_tamp = sum(1 for _, l in train_ds.samples if l == 1)
    total = n_auth + n_tamp
    weights = torch.tensor([total / (2 * n_auth), total / (2 * n_tamp)]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        build_param_groups(model, args.arch, args.lr_backbone, args.lr_new),
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.warmup, args.num_epochs)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_ds)
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        train_f1 = f1_score(train_labels, train_preds, average="macro")

        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
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
            "lr_new": scheduler.get_last_lr()[-1],
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
