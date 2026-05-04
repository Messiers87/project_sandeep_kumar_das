# =============================================================================
# Provides:
#   • predict_function  – accepts a list of file paths, returns predicted labels
#                         (also returns softmax probabilities as BONUS)
# =============================================================================

import os
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

import config
from model import GravitationalLensCNN


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str = config.checkpoint_path,
    device:          str = config.device,
) -> GravitationalLensCNN:
    """
    Instantiate GravitationalLensCNN and load weights from *checkpoint_path*.

    Parameters
    ----------
    checkpoint_path : path to the .pth file saved during training
    device          : 'cuda' or 'cpu'

    Returns
    -------
    model : GravitationalLensCNN ready for inference (eval mode, on device)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to generate weights."
        )

    model = GravitationalLensCNN(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and wrapped checkpoint dicts
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()

    print(f"✓ Loaded weights from {checkpoint_path} "
          f"(epoch {checkpoint.get('epoch', '?')})")
    return model


# ---------------------------------------------------------------------------
# Single-sample loader
# ---------------------------------------------------------------------------

def _load_sample(path: str) -> torch.Tensor:
    """
    Load a single image sample from disk.

    Supported formats
    -----------------
    • .npy  – expected shapes: (1, H, W) | (H, W) | (H, W, 1)
    • .jpg / .jpeg / .png / .bmp / .tiff  – PIL-readable

    Returns
    -------
    tensor : shape (1, resize_y, resize_x)  float32 in [0, 1]
    """
    ext = os.path.splitext(path)[1].lower()

    # ── NumPy ────────────────────────────────────────────────────────────
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False).astype(np.float32)
        if arr.max() > 1.5:          # byte range → normalise
            arr = arr / 255.0

        if arr.ndim == 2:            # (H, W) → (1, H, W)
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[-1] == 1:   # (H, W, 1) → (1, H, W)
            arr = arr.transpose(2, 0, 1)
        # else assume (1, H, W) already

        tensor = torch.from_numpy(arr)   # (1, H, W)

    # ── Image file (JPEG, PNG, …) ─────────────────────────────────────────
    else:
        img = Image.open(path).convert("L")  # grayscale
        tensor = T.ToTensor()(img)            # (1, H, W) in [0,1]

    # ── Resize to model's expected input ─────────────────────────────────
    resize = T.Resize(
        (config.resize_y, config.resize_x),
        interpolation=T.InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = resize(tensor)   # (1, resize_y, resize_x)
    return tensor


# ---------------------------------------------------------------------------
# Primary prediction function  (required by interface.py)
# ---------------------------------------------------------------------------

def predict_function(
    file_paths:      List[str],
    checkpoint_path: str = config.checkpoint_path,
    device:          str = config.device,
    return_probs:    bool = True,
) -> Union[List[int], Tuple[List[int], List[List[float]]]]:
    """
    Run inference on a list of file paths and return predicted class labels.

    Parameters
    ----------
    file_paths      : list of paths to image files (.npy | .jpg | .png | …)
    checkpoint_path : path to trained weights (.pth)
    device          : 'cuda' or 'cpu'
    return_probs    : if True, also return per-class softmax probabilities
                      (BONUS feature)

    Returns
    -------
    predicted_labels : List[int]                           – class index per sample
    probabilities    : List[List[float]]  (only if return_probs=True)
                       inner list has length num_classes, values in [0, 1]
    """
    if not file_paths:
        return ([], []) if return_probs else []

    # ── Load model ───────────────────────────────────────────────────────
    model = load_model(checkpoint_path=checkpoint_path, device=device)

    predicted_labels = []
    probabilities    = []

    # ── Inference loop ───────────────────────────────────────────────────
    with torch.no_grad():
        for path in file_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Input file not found: {path}")

            # Load and batch-ify: (1, H, W) → (1, 1, H, W)
            tensor = _load_sample(path).unsqueeze(0).to(device)

            logits = model(tensor)                       # (1, num_classes)
            probs  = F.softmax(logits, dim=1)            # (1, num_classes)
            label  = logits.argmax(dim=1).item()         # scalar int

            predicted_labels.append(int(label))
            probabilities.append(probs.squeeze(0).cpu().tolist())

            class_name = config.class_names[label]
            conf       = probs.squeeze(0)[label].item()
            print(f"  {os.path.basename(path):30s} → "
                  f"Class {label} ({class_name})  conf={conf:.4f}")

    if return_probs:
        return predicted_labels, probabilities
    return predicted_labels


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict gravitational lens substructure classes"
    )
    parser.add_argument("files", nargs="+",
                        help="Paths to image files (.npy / .jpg / .png)")
    parser.add_argument("--checkpoint", default=config.checkpoint_path,
                        help="Path to trained weights (.pth)")
    parser.add_argument("--device", default=config.device)
    args = parser.parse_args()

    labels, probs = predict_function(
        file_paths=args.files,
        checkpoint_path=args.checkpoint,
        device=args.device,
        return_probs=True,
    )

    print("\n── Results ──")
    for path, lbl, prob in zip(args.files, labels, probs):
        formatted_probs = "  ".join(
            f"{config.class_names[i]}: {p:.4f}" for i, p in enumerate(prob)
        )
        print(f"{os.path.basename(path):30s}  Pred={lbl}  [{formatted_probs}]")
