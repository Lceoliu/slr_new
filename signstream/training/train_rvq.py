from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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


def compute_codebook_usage(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    """Return codebook usage rate for each level."""
    model.eval()
    levels = model.quantizer.levels
    size = model.quantizer.codebook_size
    counts = torch.zeros(levels, size, dtype=torch.long)
    with torch.no_grad():
        for batch in tqdm(loader, desc="codebook", leave=False):
            allowed = set(getattr(model, "encoders").keys())
            inputs = {k: v.to(device) for k, v in batch.items() if k in allowed}
            _, codes, _ = model(inputs)
            for stream_codes in codes.values():
                for lvl, idx in enumerate(stream_codes):
                    flat = idx.reshape(-1).cpu()
                    counts[lvl] += torch.bincount(
                        flat, minlength=size
                    )
    usage = (counts > 0).sum(dim=1).float() / size
    return usage


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

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_epoch(model, loader, optim, device)
        val_loss = val_epoch(model, val_loader, device)

        logger.log_scalar("Loss/train", train_loss, epoch)
        logger.log_scalar("Loss/val", val_loss, epoch)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "optimizer": optim.state_dict()},
                ckpt_dir / "best.pt",
            )
            torch.save(model.quantizer.state_dict(), ckpt_dir / "best_codebook.pt")

        if (epoch + 1) % 5 == 0:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "optimizer": optim.state_dict()},
                ckpt_dir / f"epoch_{epoch+1}.pt",
            )

        if (epoch + 1) % 10 == 0:
            usage = compute_codebook_usage(model, val_loader, device)
            for lvl, rate in enumerate(usage):
                logger.log_scalar(f"CodebookUsage/level_{lvl}", rate.item(), epoch)
            print(f"codebook usage epoch {epoch+1}: {usage.tolist()}")

        logger.flush()
        print(f"epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    main(parser.parse_args())
