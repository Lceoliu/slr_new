from __future__ import annotations

import torch
import torch.nn as nn


class PoseEncoder(nn.Module):
    """Simple MLP encoder for pose streams."""

    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, J, C = x.shape
        flat = x.view(B * T, J * C)
        z = self.net(flat)
        return z.view(B, T, -1)
