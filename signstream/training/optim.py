"""Optimizer utilities."""
from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW


def build_optimizer(params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float = 0.0) -> AdamW:
    return AdamW(list(params), lr=lr, weight_decay=weight_decay)
