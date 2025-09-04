"""Transformer-based pose decoder."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import PositionalEncoding


class PoseDecoder(nn.Module):
    """Decode latent sequences back to pose space using a Transformer."""

    def __init__(self, num_joints: int, latent_dim: int, num_layers: int = 2, n_heads: int = 8) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(latent_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=n_heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(latent_dim, num_joints * 3)
        self.num_joints = num_joints

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # z: [B, T, D]
        z = self.pos_enc(z)
        h = self.decoder(z)
        out = self.output_proj(h)
        return out.view(z.size(0), z.size(1), self.num_joints, 3)
