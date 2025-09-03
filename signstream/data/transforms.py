from __future__ import annotations

import numpy as np
from typing import Dict


def normalize(stream: np.ndarray) -> np.ndarray:
    """Normalize coordinates to zero mean and unit variance per sequence."""
    mean = stream.mean(axis=(0, 1), keepdims=True)
    std = stream.std(axis=(0, 1), keepdims=True) + 1e-6
    return (stream - mean) / std


def apply_normalize(sample: Dict[str, np.ndarray], keys: list[str]) -> Dict[str, np.ndarray]:
    for k in keys:
        sample[k] = normalize(sample[k])
    return sample
