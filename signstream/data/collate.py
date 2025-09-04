"""Custom collate functions for DataLoader."""
from __future__ import annotations

from typing import Dict, List

import torch


def simple_collate(batch: List[Dict]) -> Dict:
    """Stack tensors from the batch into a single dictionary.

    The batch items are expected to be dictionaries produced by
    ``CSLDailyDataset`` where every value except ``name`` is a tensor with the
    same shape.
    """
    names = [item["name"] for item in batch]
    result: Dict[str, torch.Tensor] = {"name": names}
    keys = [k for k in batch[0].keys() if k != "name"]
    for k in keys:
        tensors = [item[k] for item in batch]
        result[k] = torch.stack(tensors, dim=0)
    return result
