"""Simple pose encoder."""
from __future__ import annotations

import torch
import torch.nn as nn


class PoseEncoder(nn.Module):
    """Encode pose sequences into latent representations."""

    def __init__(self, num_joints: int, latent_dim: int) -> None:
        super().__init__()
        in_dim = num_joints * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, J, 3]
        b, t, j, c = x.shape
        x_flat = x.view(b * t, j * c)
        out = self.net(x_flat)
        return out.view(b, t, -1)
