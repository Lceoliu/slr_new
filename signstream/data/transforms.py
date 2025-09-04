"""Data transformation utilities for CSL-Daily poses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import numpy as np
import torch

# Slices for different body parts (0-based indexing, 134 points total)
BODY_SLICE = slice(1, 24)  # 1-23, torso (foot keys ignored)
FACE_SLICE = slice(24, 92)
LEFT_HAND_SLICE = slice(92, 113)
RIGHT_HAND_SLICE = slice(113, 134)
FULL_BODY_SLICE = slice(1, 134)

# Centre indices for local coordinate systems
BODY_CENTER = 1
FACE_CENTER = 32
LEFT_HAND_CENTER = 92
RIGHT_HAND_CENTER = 113

# Keypoint pairs used for normalisation scale (left/right eyelids)
EYE_L = (66, 69)
EYE_R = (60, 63)


def _compute_scale(pose: np.ndarray) -> float:
    """Compute a global scale based on eye lid lengths.

    The scale is the average of left/right eye lid lengths across all frames.
    If the computed value is non-positive, ``1.0`` is returned.
    """

    left = np.linalg.norm(pose[:, EYE_L[0], :2] - pose[:, EYE_L[1], :2], axis=-1)
    right = np.linalg.norm(pose[:, EYE_R[0], :2] - pose[:, EYE_R[1], :2], axis=-1)
    scale = float(np.mean((left + right) / 2.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return scale


def split_and_normalise(pose: np.ndarray) -> Dict[str, np.ndarray]:
    """Split pose array into sub streams and apply normalisation.

    Each stream is translated so its designated centre keypoint becomes the
    origin and is scaled by the average eye-lid length.

    Args:
        pose: numpy array of shape ``[T, 134, 3]``

    Returns:
        dict mapping stream name to array of shape ``[T, J, 3]``.
    """

    scale = _compute_scale(pose)

    def _normalise(slice_obj: slice, centre_idx: int) -> np.ndarray:
        sub = pose[:, slice_obj].copy()
        centre = pose[:, centre_idx, :2][:, None, :]
        sub_xy = (sub[..., :2] - centre) / scale
        return np.concatenate([sub_xy, sub[..., 2:3]], axis=-1)

    return {
        "face": _normalise(FACE_SLICE, FACE_CENTER),
        "left_hand": _normalise(LEFT_HAND_SLICE, LEFT_HAND_CENTER),
        "right_hand": _normalise(RIGHT_HAND_SLICE, RIGHT_HAND_CENTER),
        "body": _normalise(BODY_SLICE, BODY_CENTER),
        "full_body": _normalise(FULL_BODY_SLICE, BODY_CENTER),
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
    return Compose([split_and_normalise, ToTensor()])
