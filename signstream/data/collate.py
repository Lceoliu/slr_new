from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    names = []
    for sample in batch:
        names.append(sample["name"])
    out["name"] = names
    for key in batch[0].keys():
        if key == "name":
            continue
        arr = np.stack([b[key] for b in batch], axis=0)
        out[key] = torch.from_numpy(arr).float()
    return out
