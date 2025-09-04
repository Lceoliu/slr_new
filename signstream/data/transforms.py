from __future__ import annotations

import numpy as np
from typing import Dict, List, Callable


def normalize(stream: np.ndarray) -> np.ndarray:
    """Normalize coordinates to zero mean and unit variance per sequence."""
    mean = stream.mean(axis=(0, 1), keepdims=True)
    std = stream.std(axis=(0, 1), keepdims=True) + 1e-6
    return (stream - mean) / std


def apply_normalize(
    sample: Dict[str, np.ndarray], keys: list[str]
) -> Dict[str, np.ndarray]:
    for k in keys:
        sample[k] = normalize(sample[k])
    return sample


def chunk_sequence(seq: np.ndarray, chunk_len: int) -> np.ndarray:
    """Split sequence into chunks of length ``chunk_len``.

    The last chunk is dropped if it is shorter than ``chunk_len``.
    """

    T = seq.shape[0]
    num_chunks = T // chunk_len
    if num_chunks == 0:
        return np.empty((0, chunk_len, *seq.shape[1:]), dtype=seq.dtype)
    return seq[: num_chunks * chunk_len].reshape(num_chunks, chunk_len, *seq.shape[1:])


class ChunkAndSplit:
    """Transform that chunks all streams in a sample."""

    def __init__(self, chunk_len: int):
        self.chunk_len = chunk_len

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {}
        for key, arr in sample.items():
            if key == "name":
                out[key] = arr
                continue
            out[key] = chunk_sequence(arr, self.chunk_len)
        return out


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
