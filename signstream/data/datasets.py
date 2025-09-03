from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
from torch.utils.data import Dataset


class CSLDailyDataset(Dataset):
    """Minimal dataset for the CSL-Daily pose data.

    Parameters
    ----------
    root: str or Path
        Root directory of the dataset containing ``frames_512x512`` and
        ``split_files.json``.
    split: str
        One of ``"train"``, ``"val"`` or ``"test"``.
    transform: callable, optional
        Optional transform applied to the loaded sample.
    """

    # index ranges for different body parts according to README
    FACE = slice(24, 92)
    LEFT_HAND = slice(92, 113)
    RIGHT_HAND = slice(113, 134)
    BODY = slice(1, 18)

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        split_file = self.root / "split_files.json"
        if not split_file.exists():
            raise FileNotFoundError(f"split file {split_file} not found")
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if split not in data:
            raise KeyError(f"split '{split}' not found in split_files.json")
        self.names: List[str] = data[split]
        self.frames_dir = self.root / "frames_512x512"

    def __len__(self) -> int:
        return len(self.names)

    def _load_pose(self, name: str) -> np.ndarray:
        path = self.frames_dir / f"{name}.npy"
        arr = np.load(path)  # [T, 134, 3]
        # drop width/height at index 0
        return arr[:, 1:, :]

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        name = self.names[idx]
        pose = self._load_pose(name)
        sample = {
            "face": pose[:, self.FACE, :2],
            "left_hand": pose[:, self.LEFT_HAND, :2],
            "right_hand": pose[:, self.RIGHT_HAND, :2],
            "body": pose[:, self.BODY, :2],
        }
        sample["full_body"] = pose[:, :, :2]
        if self.transform is not None:
            sample = self.transform(sample)
        sample["name"] = name
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
