from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    """Simple multi-level residual vector quantizer with EMA updates."""

    def __init__(
        self,
        dim: int,
        levels: int,
        codebook_size: int,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
        usage_reg: float = 0.0,
    ):
        super().__init__()
        self.levels = levels
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = commitment_beta
        self.ema_decay = ema_decay
        self.usage_reg = usage_reg

        # Initialize codebooks and EMA states
        self.codebooks = nn.ParameterList([])
        self.register_buffer("ema_cluster_size", torch.zeros(levels, codebook_size))
        self.register_buffer("ema_w", torch.randn(levels, codebook_size, dim))

        for i in range(levels):
            # Initialize codebook with random values
            self.codebooks.append(nn.Parameter(torch.randn(codebook_size, dim)))
            # Initialize EMA states for each level
            self.ema_cluster_size[i] = torch.zeros(codebook_size)
            self.ema_w[i] = torch.randn(codebook_size, dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Quantize ``x`` and return quantized tensor, code indices and loss."""
        residual = x
        quantized_out = torch.zeros_like(x)
        all_codes: List[torch.Tensor] = []
        total_loss = 0.0

        for level_idx in range(self.levels):
            codebook = self.codebooks[level_idx]
            ema_cluster_size = self.ema_cluster_size[level_idx]
            ema_w = self.ema_w[level_idx]

            flat_residual = residual.reshape(-1, self.dim)

            # Calculate distances
            distances = (
                flat_residual.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat_residual @ codebook.t()
                + codebook.pow(2).sum(dim=1)
            )

            # Find closest codebook entries
            indices = distances.argmin(dim=1)
            one_hot_indices = F.one_hot(indices, num_classes=self.codebook_size).float()

            # Quantize
            quantized = F.embedding(indices, codebook).view_as(residual)

            # Update EMA states (if training)
            if self.training:
                with torch.no_grad():
                    # Update ema_cluster_size
                    ema_cluster_size.data.mul_(self.ema_decay).add_(
                        1 - self.ema_decay, one_hot_indices.sum(dim=0)
                    )
                    # Update ema_w
                    flat_residual_sum = torch.matmul(one_hot_indices.t(), flat_residual)
                    ema_w.data.mul_(self.ema_decay).add_(
                        1 - self.ema_decay, flat_residual_sum
                    )

                    # Normalize ema_w and update codebook
                    n = ema_cluster_size.sum()
                    updated_ema_cluster_size = (
                        (ema_cluster_size + 1e-5) / (n + 1e-5) * n
                    )
                    codebook.data.copy_(ema_w / updated_ema_cluster_size.unsqueeze(1))

            # Calculate losses
            # Reconstruction loss
            reconstruction_loss = (residual.detach() - quantized).pow(2).mean()
            # Commitment loss
            commitment_loss = self.beta * (residual - quantized.detach()).pow(2).mean()

            # Usage regularization (entropy of usage distribution)
            usage_distribution = one_hot_indices.sum(dim=0) / one_hot_indices.sum()
            usage_loss = -(
                usage_distribution * torch.log(usage_distribution + 1e-10)
            ).sum()

            total_loss += (
                reconstruction_loss + commitment_loss + self.usage_reg * usage_loss
            )

            # Add current quantized output to total
            quantized_out = (
                quantized_out + quantized.detach()
            )  # Detach to prevent gradients flowing through previous levels

            # Update residual
            residual = (
                residual - quantized.detach()
            )  # Detach for straight-through estimator

            all_codes.append(indices.view(*residual.shape[:-1]))

        return quantized_out, all_codes, total_loss

    def dequantize(self, codes: List[torch.Tensor]) -> torch.Tensor:
        assert len(codes) == self.levels
        x = 0.0
        for level_idx, code in enumerate(codes):
            codebook = self.codebooks[level_idx]
            x = x + F.embedding(code, codebook)
        return x
