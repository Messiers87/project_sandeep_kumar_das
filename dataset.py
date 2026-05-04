# =============================================================================
# dataset.py
# CNN Pipeline for Strong Gravitational Lens Substructure Classification
# =============================================================================
# Provides:
#   • GravitationalLensDataset  – custom torch.utils.data.Dataset
#   • get_dataloader            – convenience DataLoader factory function
# =============================================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import config


# ---------------------------------------------------------------------------
# Custom Dataset
# ---------------------------------------------------------------------------

class GravitationalLensDataset(Dataset):
    """
    PyTorch Dataset for gravitational lensing .npy image files.

    Parameters
    ----------
    data_path   : str  – absolute/relative path to X_*.npy  (N, 1, H, W)
    label_path  : str  – absolute/relative path to y_*.npy  (N,)
    transform   : callable | None – optional torchvision transform pipeline
    resize      : bool – if True, resize images to (config.resize_x, config.resize_y)
    """

    def __init__(self, data_path: str, label_path: str,
                 transform=None, resize: bool = False):
        super().__init__()

        # Validate paths
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # Load arrays from disk  (memory-mapped for large files)
        self.data   = np.load(data_path,  allow_pickle=False).astype(np.float32)
        self.labels = np.load(label_path, allow_pickle=False).astype(np.int64)

        assert len(self.data) == len(self.labels), (
            f"Data/label length mismatch: {len(self.data)} vs {len(self.labels)}"
        )

        # Pixel normalisation: raw arrays are typically in [0, 1] already;
        # divide by 255 only when the data appears to be in byte range.
        if self.data.max() > 1.5:
            self.data = self.data / 255.0

        # Optional resize using config dimensions
        self.resize = resize
        if resize:
            self.resize_transform = T.Resize(
                (config.resize_y, config.resize_x),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            self.resize_transform = None

        # User-supplied augmentation / normalisation pipeline
        self.transform = transform

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.labels)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        """
        Returns
        -------
        image : torch.Tensor  shape (C, H, W)  dtype float32
        label : torch.Tensor  scalar            dtype int64
        """
        # Convert numpy slice → torch tensor  (C, H, W)
        image = torch.from_numpy(self.data[idx])   # already (1, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Apply resize if requested
        if self.resize_transform is not None:
            image = self.resize_transform(image)

        # Apply user-supplied transform (augmentation, normalisation, …)
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# DataLoader Factory

def get_dataloader(
    data_path:   str,
    label_path:  str,
    batch_size:  int  = config.batch_size,
    shuffle:     bool = True,
    num_workers: int  = 2,
    transform         = None,
    resize:      bool = False,
    pin_memory:  bool = True,
) -> DataLoader:
    """
    Build and return a DataLoader for the given .npy dataset.

    Parameters
    ----------
    data_path   : path to X_*.npy
    label_path  : path to y_*.npy
    batch_size  : mini-batch size           (default from config)
    shuffle     : shuffle dataset each epoch
    num_workers : parallel data-loading workers
    transform   : optional torchvision transform
    resize      : resize images to config.resize_x × config.resize_y
    pin_memory  : pin host memory for faster GPU transfer

    Returns
    -------
    torch.utils.data.DataLoader
    """
    dataset = GravitationalLensDataset(
        data_path=data_path,
        label_path=label_path,
        transform=transform,
        resize=resize,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and (config.device == "cuda"),
        drop_last=False,
    )

    return loader
