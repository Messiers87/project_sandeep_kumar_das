"""
generate_samples.py  (helper script — not part of graded submission)
Generates 10 synthetic sample .jpg images per class into the data/ folder.
Run this once: python generate_samples.py
"""

import os
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

IMG_SIZE  = 150   # pixels
N_SAMPLES = 10    # per class
RNG       = np.random.default_rng(seed=42)

CLASS_NAMES = {
    0: "no_substructure",
    1: "subhalo_substructure",
    2: "vortex_substructure",
}


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------

def _gaussian_2d(size, cx, cy, sigma):
    """Return a 2-D Gaussian blob as a float32 array."""
    xs = np.arange(size)
    ys = np.arange(size)
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return g.astype(np.float32)


def make_no_substructure(size=IMG_SIZE) -> np.ndarray:
    """
    Class 0 — smooth Einstein ring + faint Gaussian lens galaxy; no clumps.
    """
    img = np.zeros((size, size), dtype=np.float32)
    cx = cy = size // 2

    # Lens galaxy (bright central blob)
    img += 0.8 * _gaussian_2d(size, cx, cy, sigma=size * 0.06)

    # Einstein ring (annulus)
    r_ring = size * 0.22
    dr     = size * 0.025
    xs = np.arange(size)
    ys = np.arange(size)
    xx, yy = np.meshgrid(xs, ys)
    r      = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring   = np.exp(-((r - r_ring) ** 2) / (2 * dr ** 2))
    img   += 0.7 * ring

    # Low-level background noise
    img += RNG.normal(0, 0.03, (size, size)).astype(np.float32)
    return np.clip(img, 0, 1)


def make_subhalo_substructure(size=IMG_SIZE) -> np.ndarray:
    """
    Class 1 — Einstein ring with small compact subhalo clumps on the arc.
    """
    img = make_no_substructure(size)
    cx = cy = size // 2

    # Add 2–4 random subhalo blobs near the Einstein ring
    n_clumps = RNG.integers(2, 5)
    for _ in range(n_clumps):
        angle   = RNG.uniform(0, 2 * np.pi)
        r_ring  = size * 0.22
        bx      = cx + r_ring * np.cos(angle) + RNG.uniform(-8, 8)
        by      = cy + r_ring * np.sin(angle) + RNG.uniform(-8, 8)
        sigma   = RNG.uniform(size * 0.015, size * 0.04)
        amp     = RNG.uniform(0.2, 0.5)
        img    += amp * _gaussian_2d(size, bx, by, sigma=sigma)

    img += RNG.normal(0, 0.03, (size, size)).astype(np.float32)
    return np.clip(img, 0, 1)


def make_vortex_substructure(size=IMG_SIZE) -> np.ndarray:
    """
    Class 2 — Einstein ring distorted by a vortex-like spiral pattern.
    """
    img = make_no_substructure(size)
    cx = cy = size // 2

    xs = np.arange(size) - cx
    ys = np.arange(size) - cy
    xx, yy = np.meshgrid(xs, ys)

    r     = np.sqrt(xx ** 2 + yy ** 2) + 1e-6
    theta = np.arctan2(yy, xx)

    # Spiral phase perturbation
    n_arms   = RNG.integers(2, 5)
    amp      = RNG.uniform(0.08, 0.18)
    spiral   = amp * np.exp(-(r - size * 0.22) ** 2 / (2 * (size * 0.06) ** 2))
    spiral  *= np.sin(n_arms * theta + r / (size * 0.04))
    img     += spiral.astype(np.float32)

    img += RNG.normal(0, 0.03, (size, size)).astype(np.float32)
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------------------
# Generator dispatch
# ---------------------------------------------------------------------------

GENERATORS = {
    0: make_no_substructure,
    1: make_subhalo_substructure,
    2: make_vortex_substructure,
}


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_samples():
    generated = []
    for cls_idx, cls_name in CLASS_NAMES.items():
        cls_dir = os.path.join(DATA_DIR, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        gen_fn = GENERATORS[cls_idx]
        for i in range(N_SAMPLES):
            arr      = gen_fn(IMG_SIZE)                    # float32 [0,1]
            arr_u8   = (arr * 255).clip(0, 255).astype(np.uint8)
            pil_img  = Image.fromarray(arr_u8, mode="L")  # grayscale PIL

            filename = f"class{cls_idx}_{cls_name}_sample{i+1:02d}.jpg"
            save_path = os.path.join(cls_dir, filename)
            pil_img.save(save_path, quality=95)
            generated.append(save_path)
            print(f"  Saved: {save_path}")

    print(f"\n✓ Generated {len(generated)} sample images in {DATA_DIR}/")


if __name__ == "__main__":
    generate_samples()
