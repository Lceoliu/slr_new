"""Decoder symmetric to the pose encoder."""
from __future__ import annotations

import torch
import torch.nn as nn


class PoseDecoder(nn.Module):
    def __init__(self, num_joints: int, latent_dim: int) -> None:
        super().__init__()
        out_dim = num_joints * 3
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, out_dim),
        )
        self.num_joints = num_joints

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # z: [B, T, D]
        b, t, d = z.shape
        z_flat = z.view(b * t, d)
        out = self.net(z_flat)
        return out.view(b, t, self.num_joints, 3)
