import json
import numpy as np
import tempfile
from pathlib import Path

from signstream.data.datasets import CSLDailyDataset


def create_dummy_dataset(tmpdir: Path):
    root = tmpdir
    (root / "frames_512x512").mkdir()
    pose = np.zeros((2, 134, 3), dtype=np.float32)
    np.save(root / "frames_512x512" / "sample.npy", pose)
    with open(root / "split_files.json", "w") as f:
        json.dump({"train": ["sample"]}, f)
    return root


def test_dataset_loading():
    with tempfile.TemporaryDirectory() as d:
        root = create_dummy_dataset(Path(d))
        ds = CSLDailyDataset(str(root), split="train")
        assert len(ds) == 1
        item = ds[0]
        assert "full_body" in item
        assert item["full_body"].shape == (2, 133, 3)
