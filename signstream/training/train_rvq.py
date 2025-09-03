from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from signstream.data.datasets import CSLDailyDataset, ChunkAndSplit
from signstream.data.collate import collate_fn
from signstream.models.rvq.rvq_model import RVQModel
from .optim import create_optimizer
from .loop_rvq import train_epoch
from .seed import set_seed


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = torch.device(cfg["train"].get("device", "cpu"))

    transform = ChunkAndSplit(cfg["data"]["chunk_len"])
    dataset = CSLDailyDataset(cfg["data"]["root"], cfg["data"].get("split", "train"), transform=transform)
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_fn)

    joint_dims = {
        "face": 68,
        "left_hand": 21,
        "right_hand": 21,
        "body": 17,
    }
    model = RVQModel(joint_dims, cfg["model"]["latent_dim"], cfg["model"]["rvq"]["levels"], cfg["model"]["rvq"]["codebook_size"], cfg["model"]["rvq"].get("commitment_beta", 0.25))
    model.to(device)

    optim = create_optimizer(model, cfg["train"]["lr"], cfg["train"].get("wd", 0.0))

    for epoch in range(cfg["train"]["epochs"]):
        loss = train_epoch(model, loader, optim, device)
        print(f"epoch {epoch}: loss={loss:.4f}")

    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    main(parser.parse_args())
