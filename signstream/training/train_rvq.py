from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Ensure the repository root (the directory that contains 'signstream') is on sys.path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from signstream.data.datasets import CSLDailyDataset
from signstream.data.collate import collate_fn
from signstream.data.transforms import apply_normalize, ChunkAndSplit, Compose
from signstream.models.rvq.rvq_model import RVQModel
from signstream.training.optim import create_optimizer
from signstream.training.loop_rvq import train_epoch, val_epoch
from signstream.training.seed import set_seed
from signstream.training.utils_tb import TBLogger


def load_config(path: str) -> dict:
    p = Path(path)
    candidates = [p]
    if not p.is_absolute():
        # try relative to current working directory
        candidates.append(Path.cwd() / p)
        # try relative to signstream/configs directory
        candidates.append(_THIS_FILE.parents[1] / "configs" / p.name)
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        f"Config file not found. Tried: {', '.join(str(c) for c in candidates)}"
    )


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    device = torch.device(cfg["train"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = torch.device("cpu")

    # Normalize each stream, then chunk sequences
    transform = Compose(
        [
            lambda s: apply_normalize(
                s, ["face", "left_hand", "right_hand", "body", "full_body"]
            ),
            ChunkAndSplit(cfg["data"]["chunk_len"]),
        ]
    )

    dataset = CSLDailyDataset(
        cfg["data"]["root"], cfg["data"].get("split", "train"), transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=int(os.getenv("NUM_WORKERS", "2")),
        pin_memory=torch.cuda.is_available(),
    )

    # Create validation dataset and loader
    val_dataset = CSLDailyDataset(cfg["data"]["root"], "val", transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(os.getenv("NUM_WORKERS", "2")),
        pin_memory=torch.cuda.is_available(),
    )

    joint_dims = {
        "face": 68,
        "left_hand": 21,
        "right_hand": 21,
        "body": 17,
    }

    model = RVQModel(
        joint_dims,
        cfg["model"]["latent_dim"],
        cfg["model"]["rvq"]["levels"],
        cfg["model"]["rvq"]["codebook_size"],
        cfg["model"]["rvq"].get("commitment_beta", 0.25),
        ema_decay=cfg["model"]["rvq"].get("ema_decay", 0.99),
        usage_reg=cfg["model"]["rvq"].get("usage_reg", 0.0),
    )
    model.to(device)

    optim = create_optimizer(model, cfg["train"]["lr"], cfg["train"].get("wd", 0.0))

    logger = TBLogger(cfg["train"].get("log_dir", "runs/rvq_training"))

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_epoch(model, loader, optim, device)
        val_loss = val_epoch(model, val_loader, device)

        logger.log_scalar("Loss/train", train_loss, epoch)
        logger.log_scalar("Loss/val", val_loss, epoch)
        logger.flush()

        print(f"epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    main(parser.parse_args())
