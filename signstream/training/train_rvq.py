"""Entry point for training the RVQ autoencoder."""
from __future__ import annotations

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from signstream.data.datasets import CSLDailyDataset
from signstream.data.collate import simple_collate
from signstream.models.rvq.rvq_model import RVQAutoEncoder
from .seed import set_seed
from .optim import build_optimizer
from .loop_rvq import train_one_epoch
from .utils_tb import TensorBoardLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RVQ autoencoder")
    parser.add_argument("--config", type=str, default="signstream/configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    set_seed(train_cfg.get("seed", 42))

    dataset = CSLDailyDataset(cfg["csl_daily_root"], split="train", max_samples=train_cfg.get("max_samples", 10))
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=simple_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RVQAutoEncoder(133, cfg["model"]["latent_dim"], cfg["model"]["rvq"]).to(device)
    optimizer = build_optimizer(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg.get("wd", 0.0))
    logger = TensorBoardLogger("runs")

    for epoch in range(train_cfg["epochs"]):
        train_one_epoch(model, dataloader, optimizer, device, epoch, logger)

    logger.close()
    torch.save({"model": model.state_dict()}, "rvq_model.pt")


if __name__ == "__main__":
    main()
