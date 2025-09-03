from __future__ import annotations

import torch


def create_optimizer(model: torch.nn.Module, lr: float, wd: float = 0.0) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
