# =============================================================================

# This file exposes a unified, alias-stable API that the auto-grader depends on.
# Every symbol is imported under its required alias exactly as specified in the
# project submission requirements.
# =============================================================================

# ── Model ──────────────────────────────────────────────────────────────────
from model import GravitationalLensCNN as TheModel

# ── Trainer ────────────────────────────────────────────────────────────────
from train import train_model as the_trainer

# ── Predictor ──────────────────────────────────────────────────────────────
from predict import predict_function as the_predictor

# ── Dataset ────────────────────────────────────────────────────────────────
from dataset import GravitationalLensDataset as TheDataset
from dataset import get_dataloader as the_dataloader

# ── Hyperparameters ────────────────────────────────────────────────────────
from config import batch_size as the_batch_size
from config import epochs as total_epochs

# =============================================================================
# Additional exports (available for convenience but not graded)
# =============================================================================
from config import learning_rate, num_classes, input_channels, resize_x, resize_y
from config import device, checkpoint_path, class_names
from model  import GravitationalLensResNet          # BONUS: transfer-learning model
from train  import evaluate_model                   # BONUS: metrics helper
from predict import load_model                      # weight-loading utility

# =============================================================================
# Quick interface sanity check  (run as script to verify all imports resolve)
# =============================================================================

if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("  interface.py — auto-grader contract verification")
    print("=" * 60)

    # Verify model instantiation
    model = TheModel()
    dummy = torch.zeros(2, input_channels, resize_y, resize_x)
    out   = model(dummy)
    print(f"  TheModel          : {TheModel.__name__}  ✓")
    print(f"  Forward pass      : input {tuple(dummy.shape)} → output {tuple(out.shape)}  ✓")

    # Verify aliases resolve to callables
    print(f"  the_trainer       : {the_trainer.__name__}  ✓")
    print(f"  the_predictor     : {the_predictor.__name__}  ✓")
    print(f"  TheDataset        : {TheDataset.__name__}  ✓")
    print(f"  the_dataloader    : {the_dataloader.__name__}  ✓")

    # Verify hyperparameter aliases
    print(f"  the_batch_size    : {the_batch_size}  ✓")
    print(f"  total_epochs      : {total_epochs}  ✓")

    print("=" * 60)
    print("  All interface checks PASSED ✓")
    print("=" * 60)
