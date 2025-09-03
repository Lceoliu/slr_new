import json
import json
from pathlib import Path

import numpy as np

from signstream.data.datasets import CSLDailyDataset, ChunkAndSplit


def create_dataset(tmp: Path) -> Path:
    frames = tmp / "frames_512x512"
    frames.mkdir()
    arr = np.random.rand(20, 134, 3).astype("float32")
    np.save(frames / "a.npy", arr)
    with open(tmp / "split_files.json", "w", encoding="utf-8") as f:
        json.dump({"train": ["a"]}, f)
    return tmp


def test_dataset_loading(tmp_path):
    root = create_dataset(tmp_path)
    ds = CSLDailyDataset(root, transform=ChunkAndSplit(5))
    sample = ds[0]
    assert sample["face"].shape[0] == 4  # num_chunks
    assert sample["left_hand"].shape[2] == 21
    assert sample["left_hand"].shape[3] == 2
