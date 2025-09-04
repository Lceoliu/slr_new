"""Transformer-based pose encoder."""
"""Transformer-based pose encoder."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 1000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PoseEncoder(nn.Module):
    """Encode pose sequences into latent representations using a Transformer."""

    def __init__(self, num_joints: int, latent_dim: int, num_layers: int = 2, n_heads: int = 8) -> None:
        super().__init__()
        in_dim = num_joints * 3
        self.input_proj = nn.Linear(in_dim, latent_dim)
        self.pos_enc = PositionalEncoding(latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, J, 3]
        b, t, j, c = x.shape
        x_flat = x.view(b, t, j * c)
        x_emb = self.input_proj(x_flat)
        x_emb = self.pos_enc(x_emb)
        return self.encoder(x_emb)
