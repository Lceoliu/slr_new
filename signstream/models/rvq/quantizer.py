from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class ResidualVectorQuantizer(nn.Module):
    """Simple multi-level residual vector quantizer."""

    def __init__(self, dim: int, levels: int, codebook_size: int, commitment_beta: float = 0.25):
        super().__init__()
        self.levels = nn.ModuleList([nn.Embedding(codebook_size, dim) for _ in range(levels)])
        self.beta = commitment_beta
        self.codebook_size = codebook_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Quantize ``x`` and return quantized tensor, code indices and loss."""
        residual = x
        quantized = 0.0
        codes: List[torch.Tensor] = []
        loss = 0.0
        for emb in self.levels:
            flat = residual.reshape(-1, residual.shape[-1])
            distances = (
                flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat @ emb.weight.t()
                + emb.weight.pow(2).sum(dim=1)
            )
            indices = distances.argmin(dim=1)
            q = emb(indices).view_as(residual)
            codes.append(indices.view(*residual.shape[:-1]))
            quantized = quantized + q
            loss = loss + (residual.detach() - q).pow(2).mean() + self.beta * (residual - q.detach()).pow(2).mean()
            residual = residual - q
        return quantized, codes, loss

    def dequantize(self, codes: List[torch.Tensor]) -> torch.Tensor:
        assert len(codes) == len(self.levels)
        x = 0.0
        for code, emb in zip(codes, self.levels):
            x = x + emb(code)
        return x
