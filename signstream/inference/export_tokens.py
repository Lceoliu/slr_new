from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from signstream.data.datasets import CSLDailyDataset, ChunkAndSplit
from signstream.models.rvq.rvq_model import RVQModel
from signstream.inference.rle import run_length_encode


STREAM_ABBR = {
    "face": "F",
    "left_hand": "LH",
    "right_hand": "RH",
    "body": "B",
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def codes_to_tokens(codes: Dict[str, List[torch.Tensor]]) -> Dict[str, List[List[int]]]:
    """Convert code tensors to Python lists."""
    result: Dict[str, List[List[int]]] = {}
    for stream, levels in codes.items():
        T = levels[0].shape[1]
        stream_tokens: List[List[int]] = []
        for t in range(T):
            stream_tokens.append([int(level[0, t]) for level in levels])
        result[stream] = stream_tokens
    return result


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = torch.device("cpu")
    transform = ChunkAndSplit(cfg["data"]["chunk_len"])
    dataset = CSLDailyDataset(cfg["data"]["root"], args.split, transform=transform)

    joint_dims = {
        "face": 68,
        "left_hand": 21,
        "right_hand": 21,
        "body": 17,
    }
    model = RVQModel(joint_dims, cfg["model"]["latent_dim"], cfg["model"]["rvq"]["levels"], cfg["model"]["rvq"]["codebook_size"], cfg["model"]["rvq"].get("commitment_beta", 0.25))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(min(args.num_samples, len(dataset))):
            sample = dataset[i]
            inputs = {k: torch.from_numpy(v).unsqueeze(0).float().to(device) for k, v in sample.items() if k != "name"}
            with torch.no_grad():
                _, codes, _ = model(inputs)
            tokens = codes_to_tokens(codes)
            if cfg["export"].get("enable_rle", False):
                tokens = {k: run_length_encode(v) for k, v in tokens.items()}
            out = {
                "video_id": sample["name"],
                "tokens": tokens,
                "rle": cfg["export"].get("enable_rle", False),
            }
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--out", type=str, required=True)
    main(parser.parse_args())
