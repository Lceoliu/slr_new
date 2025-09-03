from __future__ import annotations

import torch


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - y) ** 2)
