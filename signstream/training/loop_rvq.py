from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses import reconstruction_and_quant_loss


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        # Only pass streams that the model has encoders for
        allowed = set(getattr(model, "encoders").keys())
        inputs: Dict[str, torch.Tensor] = {
            k: v.to(device) for k, v in batch.items() if k in allowed
        }
        optimizer.zero_grad()
        outputs, codes, q_loss = model(inputs)
        loss = reconstruction_and_quant_loss(inputs, outputs, q_loss)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    return total / max(1, len(loader))


def val_epoch(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            allowed = set(getattr(model, "encoders").keys())
            inputs: Dict[str, torch.Tensor] = {
                k: v.to(device) for k, v in batch.items() if k in allowed
            }
            outputs, codes, q_loss = model(inputs)
            loss = reconstruction_and_quant_loss(inputs, outputs, q_loss)
            total += float(loss.detach())
    return total / max(1, len(loader))
