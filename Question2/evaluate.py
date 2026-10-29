"""Evaluate models on the 1,000 test videos and save per-video results.

Usage: python evaluate.py Blank CopyLast A [B C D symB symD ...] [--original-only]
Learned models are evaluated for seeds 0 and 1 from runs/<model>_s<seed>.pt.
"sym<model>" wraps the trained model as a symmetrised forecaster.
Writes results/eval/<name>[_s<seed>].pt with per-video tensors, inference time and parameters.
"""
import argparse
from pathlib import Path

import torch

from q2 import data, metrics, models

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "eval"
SEEDS = (0, 1)


def load(name, seed, device):
    symmetrised = name.startswith("sym")
    base = name[3:] if symmetrised else name
    model = models.build(base)
    model.load_state_dict(torch.load(HERE / "runs" / f"{base}_s{seed}.pt", map_location="cpu"))
    model = model.to(device).eval()
    return models.Symmetrised(model) if symmetrised else model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--original-only", action="store_true",
                        help="only original 1x clips (Table 2, before the Q2.5 commit)")
    args = parser.parse_args()

    device = torch.device("cuda")
    test = data.load_videos("test")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        if name in ("Blank", "CopyLast"):
            jobs = [(name, None, models.Blank() if name == "Blank" else models.CopyLast())]
        else:
            jobs = [(f"{name}_s{seed}", seed, load(name, seed, device)) for seed in SEEDS]
        for key, seed, model in jobs:
            results = metrics.evaluate(model, test, original_only=args.original_only)
            results["parameters"] = models.count_parameters(model)
            if seed is not None:
                results["inference_seconds"] = metrics.inference_seconds(model, test, device)
            torch.save(results, OUT / f"{key}.pt")
            summary = {k: round(v.mean().item(), 6) for k, v in results.items() if torch.is_tensor(v) and v.dim() == 1}
            print(key, summary, flush=True)


if __name__ == "__main__":
    main()
