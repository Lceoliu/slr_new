"""Export discrete tokens from a trained model."""
from __future__ import annotations

import argparse
import json
import yaml
import torch

from signstream.data.datasets import CSLDailyDataset
from signstream.data.collate import simple_collate
from signstream.models.rvq.rvq_model import RVQAutoEncoder
from .rle import run_length_encode


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tokens")
    parser.add_argument("--config", type=str, default="signstream/configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="tokens.jsonl")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = CSLDailyDataset(cfg["csl_daily_root"], split="val", max_samples=args.num_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=simple_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RVQAutoEncoder(133, cfg["model"]["latent_dim"], cfg["model"]["rvq"]).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    with open(args.out, "w", encoding="utf-8") as out_f:
        for batch in loader:
            name = batch["name"][0]
            x = batch["full_body"].to(device)
            with torch.no_grad():
                out = model(x)
            indices = [ind.squeeze(0).tolist() for ind in out["indices"]]
            tokens = {
                "video_name": name,
                "tokens": indices,
            }
            out_f.write(json.dumps(tokens) + "\n")


if __name__ == "__main__":
    main()
