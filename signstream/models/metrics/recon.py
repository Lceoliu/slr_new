"""Reconstruction metrics."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error."""
    return F.mse_loss(recon, target)
