"""Training loop for the RVQ autoencoder."""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

from .utils_tb import TensorBoardLogger


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    logger: Optional[TensorBoardLogger] = None,
) -> None:
    model.train()
    for step, batch in enumerate(dataloader):
        x = batch["full_body"].to(device)
        out = model(x)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if logger is not None:
            global_step = epoch * len(dataloader) + step
            logger.log(global_step, loss=loss.item(), recon=out["recon_loss"].item(), q=out["q_loss"].item())
