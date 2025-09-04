from __future__ import annotations

import torch


def create_optimizer(
    model: torch.nn.Module, lr: float | str, wd: float | str = 0.0
) -> torch.optim.Optimizer:
    lr_f = float(lr)
    wd_f = float(wd)
    return torch.optim.AdamW(model.parameters(), lr=lr_f, weight_decay=wd_f)
