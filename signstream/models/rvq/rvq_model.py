"""End-to-end RVQ autoencoder model."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PoseEncoder
from .decoder import PoseDecoder
from .quantizer import ResidualVectorQuantizer


class RVQAutoEncoder(nn.Module):
    """Shared encoder/decoder with a residual vector quantiser."""

    def __init__(self, num_joints: int, latent_dim: int, rvq_cfg: Dict) -> None:
        super().__init__()
        self.encoder = PoseEncoder(num_joints, latent_dim)
        self.quantizer = ResidualVectorQuantizer(
            rvq_cfg.get("levels", 2),
            rvq_cfg.get("codebook_size", 512),
            latent_dim,
            rvq_cfg.get("ema_decay", 0.99),
        )
        self.decoder = PoseDecoder(num_joints, latent_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.encoder(x)
        quantized, indices, q_loss = self.quantizer(latent)
        recon = self.decoder(quantized)
        recon_loss = F.mse_loss(recon, x)
        total_loss = recon_loss + q_loss
        return {
            "recon": recon,
            "indices": indices,
            "loss": total_loss,
            "recon_loss": recon_loss,
            "q_loss": q_loss,
        }
