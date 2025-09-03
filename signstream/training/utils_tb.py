from __future__ import annotations

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


class TBLogger:
    def __init__(self, log_dir: str | None = None):
        self.writer = SummaryWriter(log_dir) if SummaryWriter is not None else None

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()
