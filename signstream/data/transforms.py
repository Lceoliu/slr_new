"""Data transformation utilities for CSL-Daily poses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Callable
import numpy as np
import torch

# Slices for different body parts (0-based indexing, 134 points total)
BODY_SLICE = slice(1, 24)  # 1-23, torso (foot keys ignored)
FACE_SLICE = slice(24, 92)
LEFT_HAND_SLICE = slice(92, 113)
RIGHT_HAND_SLICE = slice(113, 134)
FULL_BODY_SLICE = slice(1, 134)


def split_streams(pose: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a pose array into sub streams.

    Args:
        pose: numpy array of shape [T, 134, 3]

    Returns:
        dict mapping stream name to array of shape [T, J, 3]
    """
    return {
        "face": pose[:, FACE_SLICE],
        "left_hand": pose[:, LEFT_HAND_SLICE],
        "right_hand": pose[:, RIGHT_HAND_SLICE],
        "body": pose[:, BODY_SLICE],
        "full_body": pose[:, FULL_BODY_SLICE],
    }


@dataclass
class Compose:
    """Compose several transforms together."""
    transforms: Iterable[Callable]

    def __call__(self, pose: np.ndarray) -> Dict[str, np.ndarray]:
        result = pose
        for t in self.transforms:
            result = t(result)
        return result


class ToTensor:
    """Convert numpy arrays in the sample dictionary to torch tensors."""

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {k: torch.from_numpy(v).float() for k, v in sample.items()}


def build_default_transform() -> Compose:
    """Return the default transformation pipeline."""
    return Compose([split_streams, ToTensor()])
