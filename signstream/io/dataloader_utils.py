"""DataLoader helper functions."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
