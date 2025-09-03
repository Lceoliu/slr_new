from __future__ import annotations

import torch
import torch.nn as nn


class PoseDecoder(nn.Module):
    """Simple MLP decoder to reconstruct pose streams."""

    def __init__(self, latent_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.view(B * T, D)
        out = self.net(flat)
        return out.view(B, T, -1)
