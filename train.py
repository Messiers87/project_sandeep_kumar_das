#  =============================================================================
# Entry point for model training.
#
# Usage (CLI example):
#   python train.py \
#       --train_data  /path/to/X_train.npy \
#       --train_labels /path/to/y_train.npy \
#       --test_data   /path/to/X_test.npy  \
#       --test_labels /path/to/y_test.npy
# =============================================================================

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import config
from model import GravitationalLensCNN


# ---------------------------------------------------------------------------
# Core training function  (required by interface.py)
# ---------------------------------------------------------------------------

def train_model(
    model:        nn.Module,
    num_epochs:   int,
    train_loader: DataLoader,
    val_loader:   DataLoader = None,
    save_path:    str        = config.checkpoint_path,
    device:       str        = config.device,
) -> dict:
    """
    Train *model* for *num_epochs* using the provided DataLoader.

    Parameters
    ----------
    model        : an nn.Module (e.g. GravitationalLensCNN)
    num_epochs   : total training epochs
    train_loader : DataLoader yielding (image, label) batches
    val_loader   : optional validation DataLoader for per-epoch eval
    save_path    : where to write the final checkpoint  (.pth)
    device       : 'cuda' or 'cpu'

    Returns
    -------
    history : dict with keys
        'train_loss', 'train_acc', 'val_loss', 'val_acc'  (lists, one per epoch)
    """

    model = model.to(device)

    # ── Loss & Optimiser ─────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ── Learning-rate scheduler ──────────────────────────────────────────
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_scheduler_step,
        gamma=config.lr_scheduler_gamma,
    )

    # ── History containers ───────────────────────────────────────────────
    history = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
    }

    best_val_acc = 0.0

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ── Train phase ──────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ── Validation phase ─────────────────────────────────────────────
        val_loss = 0.0
        val_acc  = 0.0

        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Save best checkpoint based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                _save_checkpoint(model, save_path, epoch, val_acc)

        # Step LR scheduler
        scheduler.step()

        elapsed = time.time() - epoch_start
        _log_epoch(epoch, num_epochs, train_loss, train_acc,
                   val_loss, val_acc, elapsed, scheduler)

    # ── Save final weights (always) ───────────────────────────────────────
    _save_checkpoint(model, save_path, num_epochs,
                     val_acc if val_loader else train_acc)
    print(f"\n✓ Final weights saved → {save_path}")

    return history


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
) -> tuple:
    """
    Compute average loss and accuracy over *loader*.

    Returns
    -------
    (avg_loss, accuracy)  both floats in [0, 1]
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Full evaluation with classification report + confusion matrix  (BONUS)
# ---------------------------------------------------------------------------

def evaluate_model(
    model:      nn.Module,
    loader:     DataLoader,
    device:     str = config.device,
) -> dict:
    """
    Run full evaluation on *loader* and return per-class metrics.

    Returns
    -------
    dict with keys: accuracy, classification_report, confusion_matrix
    """
    model.eval()
    model = model.to(device)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc     = (all_preds == all_labels).mean()
    report  = classification_report(all_labels, all_preds,
                                    target_names=config.class_names,
                                    digits=4)
    cm      = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"{'='*60}")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("="*60)

    return {
        "accuracy":               acc,
        "classification_report":  report,
        "confusion_matrix":       cm,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model: nn.Module, path: str, epoch: int, metric: float):
    """Persist model state_dict and metadata to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "metric":      metric,
        "state_dict":  model.state_dict(),
        "model_class": model.__class__.__name__,
    }, path)


# ---------------------------------------------------------------------------
# Epoch logging
# ---------------------------------------------------------------------------

def _log_epoch(epoch, num_epochs, train_loss, train_acc,
               val_loss, val_acc, elapsed, scheduler):
    lr = scheduler.get_last_lr()[0]
    val_str = (f"  Val Loss: {val_loss:.4f}  Val Acc: {val_acc*100:.2f}%"
               if val_loss > 0 else "")
    print(
        f"Epoch [{epoch:>3}/{num_epochs}]  "
        f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%"
        f"{val_str}  LR: {lr:.6f}  Time: {elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN for Gravitational Lens Substructure Classification"
    )
    parser.add_argument("--train_data",   required=True,
                        help="Path to X_train.npy")
    parser.add_argument("--train_labels", required=True,
                        help="Path to y_train.npy")
    parser.add_argument("--test_data",    default=None,
                        help="Path to X_test.npy  (optional, used for final eval)")
    parser.add_argument("--test_labels",  default=None,
                        help="Path to y_test.npy  (optional)")
    parser.add_argument("--epochs",       type=int, default=config.epochs)
    parser.add_argument("--batch_size",   type=int, default=config.batch_size)
    parser.add_argument("--lr",           type=float, default=config.learning_rate)
    parser.add_argument("--save_path",    default=config.checkpoint_path)
    return parser.parse_args()


if __name__ == "__main__":
    import random
    from dataset import get_dataloader

    args = _parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    print(f"Device : {config.device}")
    print(f"Epochs : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")

    # ── DataLoaders ──────────────────────────────────────────────────────
    train_loader = get_dataloader(
        data_path=args.train_data,
        label_path=args.train_labels,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = None
    if args.test_data and args.test_labels:
        val_loader = get_dataloader(
            data_path=args.test_data,
            label_path=args.test_labels,
            batch_size=args.batch_size,
            shuffle=False,
        )

    # ── Model ────────────────────────────────────────────────────────────
    model = GravitationalLensCNN(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {total_params:,}")

    # ── Train ────────────────────────────────────────────────────────────
    history = train_model(
        model=model,
        num_epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=args.save_path,
        device=config.device,
    )

    # ── Final evaluation ─────────────────────────────────────────────────
    if val_loader is not None:
        print("\n── Final Evaluation on Test Set ──")
        evaluate_model(model, val_loader, device=config.device)
