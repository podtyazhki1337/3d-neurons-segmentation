"""Reading and writing (Z, Y, X) TIFF stacks."""
from __future__ import annotations

import numpy as np


def read_volume(path: str) -> np.ndarray:
    import tifffile
    v = tifffile.imread(path)
    if v.ndim == 4:  # (Z, Y, X, C) stacks: first channel
        v = v[..., 0]
    if v.ndim == 2:
        v = v[None]
    return v


def write_labels(path: str, labels: np.ndarray) -> None:
    import tifffile
    labels = np.asarray(labels)
    dtype = np.uint16 if labels.max() < 65535 else np.uint32
    tifffile.imwrite(path, labels.astype(dtype), compression="zlib")
