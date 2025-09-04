"""Residual vector quantiser with EMA updates."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAQuantizer(nn.Module):
    """A single level vector quantizer using EMA updates."""

    def __init__(self, codebook_size: int, dim: int, decay: float = 0.99, eps: float = 1e-5) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(codebook_size, dim)
        self.embedding = nn.Parameter(embed)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed", torch.zeros(codebook_size, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.reshape(-1, self.dim)
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(indices, self.embedding).view_as(x)

        if self.training:
            one_hot = F.one_hot(indices, self.codebook_size).type_as(flat)
            self.ema_cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
            self.ema_embed.mul_(self.decay).add_(one_hot.t() @ flat, alpha=1 - self.decay)
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.eps)
                / (n + self.codebook_size * self.eps)
                * n
            )
            embed_normalised = self.ema_embed / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalised)

        loss = F.mse_loss(quantized.detach(), x) + F.mse_loss(quantized, x.detach())
        quantized = x + (quantized - x).detach()
        return quantized, indices.view(x.shape[:-1]), loss


class ResidualVectorQuantizer(nn.Module):
    """Multi-level residual vector quantizer."""

    def __init__(self, levels: int, codebook_size: int, dim: int, decay: float = 0.99) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            [EMAQuantizer(codebook_size, dim, decay) for _ in range(levels)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        residual = x
        quantized_outputs = []
        indices_list: List[torch.Tensor] = []
        loss = torch.tensor(0.0, device=x.device)
        for q in self.levels:
            q_out, q_ind, q_loss = q(residual)
            quantized_outputs.append(q_out)
            indices_list.append(q_ind)
            residual = residual - q_out
            loss = loss + q_loss
        quantized = torch.stack(quantized_outputs, dim=0).sum(0)
        return quantized, indices_list, loss
