from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
from torch.utils.data import Dataset
from signstream.data.transforms import ChunkAndSplit


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
    # After dropping the 0th index (width/height), adjust indices accordingly
    FACE = slice(23, 91)  # original 24..91 -> 23..90 after drop
    LEFT_HAND = slice(91, 112)  # original 92..112 -> 91..111 after drop
    RIGHT_HAND = slice(112, 133)  # original 113..133 -> 112..132 after drop
    BODY = slice(0, 17)  # original 1..17 -> 0..16 after drop

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self._npy_index: Optional[Dict[str, Path]] = None

        split_file = self.root / "split_files.json"
        if not split_file.exists():
            raise FileNotFoundError(f"split file {split_file} not found")
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if split not in data:
            raise KeyError(f"split '{split}' not found in split_files.json")
        self.names: List[str] = data[split]
        # Heuristic to locate npy directory
        candidates = [
            self.root / "frames_512x512",
            self.root / "keypoints",
            self.root / "npy",
            self.root,
        ]
        chosen: Optional[Path] = None
        for cand in candidates:
            if cand.is_dir():
                # pick first dir that has any npy
                any_npy = next(cand.glob("*.npy"), None)
                if any_npy is not None:
                    chosen = cand
                    break
        self.frames_dir = chosen if chosen is not None else self.root

        # Filter out names that don't exist to handle split/contents mismatches gracefully
        try:
            # Prefer a quick scan in the selected frames_dir
            present_stems = (
                {p.stem for p in self.frames_dir.glob("*.npy")}
                if self.frames_dir.is_dir()
                else set()
            )
            if not present_stems:
                # Fallback to a broader scan (slower but robust)
                present_stems = {p.stem for p in self.root.rglob("*.npy")}
            before = len(self.names)
            if before:
                self.names = [n for n in self.names if n in present_stems]
            after = len(self.names)
            missing = before - after
            if missing > 0:
                print(
                    f"[CSLDailyDataset] split '{self.split}': filtered {missing} missing out of {before}. Using {after}."
                )
        except Exception:
            # Best-effort; if anything fails, keep original names and rely on lazy index in _load_pose
            pass

    def __len__(self) -> int:
        return len(self.names)

    def _load_pose(self, name: str) -> np.ndarray:
        path = self.frames_dir / f"{name}.npy"
        if not path.exists():
            # Lazy-build an index of all npy files under root (one-time cost)
            if self._npy_index is None:
                self._npy_index = {}
                for p in self.root.rglob("*.npy"):
                    self._npy_index[p.stem] = p
            # Try direct match, then contains-match
            alt = None
            if name in self._npy_index:
                alt = self._npy_index[name]
            else:
                for stem, p in self._npy_index.items():
                    if name in stem or stem in name:
                        alt = p
                        break
            if alt is None:
                raise FileNotFoundError(
                    f"Pose npy not found for '{name}'. Looked in '{self.frames_dir}'."
                )
            path = alt
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
