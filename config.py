# =============================================================================
# All hyperparameters and project-wide configuration constants are defined here.
# =============================================================================

import torch
import os

# ---------------------------------------------------------------------------
# Dataset & Class Configuration
# ---------------------------------------------------------------------------
num_classes     = 3          # 0: No substructure | 1: Subhalo | 2: Vortex
input_channels  = 1          # Grayscale images → single channel
class_names     = ["No Substructure", "Subhalo Substructure", "Vortex Substructure"]

# ---------------------------------------------------------------------------
# Image Dimensions
# ---------------------------------------------------------------------------
resize_x = 150               # Target image width  (pixels)
resize_y = 150               # Target image height (pixels)

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
batch_size      = 64         # Mini-batch size for DataLoader
epochs          = 30         # Total training epochs
learning_rate   = 1e-3       # Adam optimizer learning rate
weight_decay    = 1e-4       # L2 regularisation coefficient (Adam)

# ---------------------------------------------------------------------------
# Learning Rate Scheduler
# ---------------------------------------------------------------------------
lr_scheduler_step   = 10    # StepLR: decay LR every N epochs
lr_scheduler_gamma  = 0.5   # StepLR: multiplicative decay factor

# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
checkpoint_dir      = os.path.join(os.path.dirname(__file__), "checkpoints")
checkpoint_path     = os.path.join(checkpoint_dir, "final_weights.pth")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Random Seed (reproducibility)
# ---------------------------------------------------------------------------
seed = 42

# ---------------------------------------------------------------------------
# Data Paths  (set at runtime — NOT hardcoded)
# ---------------------------------------------------------------------------
# These are empty by default; supply them via CLI args or environment variables
# in train.py / predict.py.
default_train_data_path = ""   # e.g. "/path/to/X_train.npy"
default_train_label_path = ""  # e.g. "/path/to/y_train.npy"
default_test_data_path  = ""   # e.g. "/path/to/X_test.npy"
default_test_label_path = ""   # e.g. "/path/to/y_test.npy"
