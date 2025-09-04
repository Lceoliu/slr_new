from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    names = [sample["name"] for sample in batch]
    out["name"] = names

    # Determine the min number of chunks for truncation to allow stacking
    def min_chunks_for_key(k: str) -> int:
        return int(min(b[k].shape[0] for b in batch))

    for key in batch[0].keys():
        if key == "name":
            continue
        min_chunks = min_chunks_for_key(key)
        trimmed = [b[key][:min_chunks] for b in batch]
        arr = np.stack(trimmed, axis=0)
        out[key] = torch.from_numpy(arr).float()
    return out
