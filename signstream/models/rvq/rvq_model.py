from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .encoder import PoseEncoder
from .decoder import PoseDecoder
from .quantizer import ResidualVectorQuantizer


class RVQModel(nn.Module):
    """Encoder -> RVQ -> Decoder for multiple pose streams."""

    def __init__(
        self,
        joint_dims: Dict[str, int],
        latent_dim: int,
        levels: int,
        codebook_size: int,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
        usage_reg: float = 0.0,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {k: PoseEncoder(v * 2, latent_dim) for k, v in joint_dims.items()}
        )
        self.decoders = nn.ModuleDict(
            {k: PoseDecoder(latent_dim, v * 2) for k, v in joint_dims.items()}
        )
        self.quantizer = ResidualVectorQuantizer(
            latent_dim, levels, codebook_size, commitment_beta, ema_decay, usage_reg
        )

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        codes: Dict[str, List[torch.Tensor]] = {}
        total_loss = 0.0
        for key, x in inputs.items():
            # inputs x: [B, Nc, T, J, C] from collate (Nc chunks)
            B, Nc, T, J, C = x.shape
            x_flat = x.view(B * Nc, T, J, C)
            z = self.encoders[key](x_flat)  # [B*Nc, T, D]
            q, idxs, q_loss = self.quantizer(z)
            recon = self.decoders[key](q).view(B, Nc, T, J * C)
            recon = recon.view_as(x)
            outputs[key] = recon
            codes[key] = idxs
            total_loss = total_loss + q_loss + (recon - x).pow(2).mean()
        return outputs, codes, total_loss
