"""
Training ResNet-18 on NEU-DET with validation, metrics and checkpointing.
"""
import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.config import (
    NUM_CLASSES,
    CLASSES,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
    DEVICE,
)
from train.dataset import get_dataloaders


# ---------- Reproducibility ----------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- Model ----------
def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Load pretrained ResNet-18 and adapt classifier."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze earlier layers (fine-tune only classifier)
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last block (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final fc layer for 6 classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


# ---------- Training ----------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Single training epoch."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Model validation."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="weighted")

    return val_loss, val_acc, val_f1, np.array(all_labels), np.array(all_preds)


# ---------- Plots ----------
def plot_metrics(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_dir: Path,
):
    """Save loss and accuracy plots."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-o", label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, train_accs, "b-o", label="Train Acc")
    ax2.plot(epochs, val_accs, "r-o", label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / "training_metrics.png", dpi=150)
    print(f"  Plots saved to {save_dir / 'training_metrics.png'}")
    plt.close()


# ---------- Main loop ----------
def main():
    set_seed()
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    print(f"Number of classes: {NUM_CLASSES} -> {CLASSES}")

    # Prepare checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    train_loader, val_loader = get_dataloaders()

    # Model
    model = build_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # History
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = 7

    print(f"\n{'='*60}")
    print(f"Starting training for {EPOCHS} epochs")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # Training
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        val_loss, val_acc, val_f1, true_labels, preds = validate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Scheduler
        scheduler.step(val_loss)

        print(
            f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}\n"
            f"  Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch

            checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "classes": CLASSES,
                },
                checkpoint_path,
            )
            print(f"  New best model! F1={val_f1:.4f} (epoch {epoch})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping after {epoch} epochs (no F1 improvement for {early_stop_patience} epochs)")
                break

    print(f"\n{'='*60}")
    print(f"Training finished. Best model: epoch {best_epoch}, F1={best_val_f1:.4f}")
    print(f"{'='*60}")

    # Save final metrics
    metrics = {
        "best_epoch": best_epoch,
        "best_val_f1": float(best_val_f1),
        "best_val_acc": float(val_accs[best_epoch - 1] if best_epoch <= len(val_accs) else 0),
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "train_accs": [float(x) for x in train_accs],
        "val_accs": [float(x) for x in val_accs],
    }
    with open(CHECKPOINT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_metrics(train_losses, val_losses, train_accs, val_accs, CHECKPOINT_DIR)

    # Confusion matrix for best model
    if best_epoch > 0:
        best_checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=device, weights_only=True)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        _, _, _, true_labels, preds = validate(model, val_loader, criterion, device)

        cm = confusion_matrix(true_labels, preds)
        print("\nConfusion matrix (validation):")
        print(f"  Classes: {CLASSES}")
        print(f"  {cm}")

        # Save confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix - Best Model")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        tick_marks = range(len(CLASSES))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(CLASSES)

        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.tight_layout()
        plt.savefig(CHECKPOINT_DIR / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"  Confusion matrix saved to {CHECKPOINT_DIR / 'confusion_matrix.png'}")

    print("\nDone!")

    # --- Automatically compute baseline for drift monitoring ---
    print("\n" + "=" * 60)
    print("Computing baseline distribution for drift monitoring...")
    print("=" * 60)
    try:
        from monitoring.data_drift import compute_baseline
        compute_baseline()
        print("Baseline computed successfully.")
    except Exception as e:
        print(f"WARNING: Could not compute baseline: {e}")
        print("You can run it manually: python -m monitoring.data_drift")

    print("\nDone!")


if __name__ == "__main__":
    main()