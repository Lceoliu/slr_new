from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from .losses import reconstruction_and_quant_loss


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        inputs: Dict[str, torch.Tensor] = {k: v.to(device) for k, v in batch.items() if k != "name"}
        optimizer.zero_grad()
        outputs, codes, q_loss = model(inputs)
        loss = reconstruction_and_quant_loss(inputs, outputs, q_loss)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    return total / max(1, len(loader))
