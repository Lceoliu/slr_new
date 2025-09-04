"""TensorBoard logging utilities."""
from __future__ import annotations

from typing import Any

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str) -> None:
        self.writer = SummaryWriter(log_dir)

    def log(self, step: int, **metrics: Any) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        self.writer.close()
