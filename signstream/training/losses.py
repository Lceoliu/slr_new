from __future__ import annotations

from typing import Dict

import torch

from signstream.models.metrics.recon import mse


def reconstruction_and_quant_loss(
    inputs: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    quant_loss: torch.Tensor,
) -> torch.Tensor:
    rec = 0.0
    for k in inputs.keys():
        rec = rec + mse(outputs[k], inputs[k])
    return rec + quant_loss
