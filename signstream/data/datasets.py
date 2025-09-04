"""Dataset definitions for SignStream project."""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from .transforms import build_default_transform


class CSLDailyDataset(Dataset):
    """Minimal dataset for the CSL-Daily pose data.

    The dataset expects the directory structure described in
    ``CSL-Daily_README.md``. Only the pose ``.npy`` files are required for
    the minimal training pipeline.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform or build_default_transform()

        split_file = os.path.join(root, "split_files.json")
        with open(split_file, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        names: List[str] = split_info.get(split, [])

        self.samples: List[Dict[str, str]] = []
        for name in names:
            npy_path = os.path.join(root, "frames_512x512", f"{name}.npy")
            if os.path.exists(npy_path):
                self.samples.append({"name": name, "npy": npy_path})
            if max_samples is not None and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        sample = self.samples[idx]
        pose = np.load(sample["npy"])  # [T, 134, 3]
        if self.transform is not None:
            data = self.transform(pose)
        else:
            data = {"full_body": pose}
        data["name"] = sample["name"]
        return data
