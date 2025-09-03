from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
from typing import Callable, Optional


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, collate_fn: Optional[Callable] = None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
