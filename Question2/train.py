"""Train one model with one seed, using the protocol in the handout.

Usage: python train.py --model A --seed 0
Saves runs/<model>_s<seed>.pt (final-epoch weights) and runs/<model>_s<seed>.json (log).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from q2 import data, metrics, models, transforms

RUNS = Path(__file__).resolve().parent / "runs"
EPOCHS = 30
BATCH = 32
LR = 1e-3


def masked_loss(prediction, target, mask):
    """Squared error averaged over pixels and available target frames, then over videos."""
    per_frame = ((prediction - target) ** 2).mean(dim=(2, 3, 4))  # [B, 5]
    per_video = (per_frame * mask).sum(dim=1) / mask.sum(dim=1)
    return per_video.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(models.CONFIGS))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    _, speed_augmented, symmetry_augmented = models.CONFIGS[args.model]

    # The seed controls initialisation, shuffling, window sampling and augmentation.
    torch.manual_seed(args.seed)
    model = models.build(args.model).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    train = data.load_videos("train").to(device)
    val = data.load_videos("val").to(device)
    val_x, val_y, _ = data.fixed_windows(val, k=1)
    val_x, val_y = data.to_float(val_x), data.to_float(val_y)

    log = {"model": args.model, "seed": args.seed, "parameters": models.count_parameters(model),
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "epoch_seconds": [], "train_loss": [], "val_E5": []}
    print(f"model {args.model} seed {args.seed}: {log['parameters']:,} parameters", flush=True)

    for epoch in range(EPOCHS):
        model.train()
        torch.cuda.synchronize()
        start = time.perf_counter()
        order = torch.randperm(len(train), generator=generator)
        total, count = 0.0, 0
        for first in range(0, len(train), BATCH):
            index = order[first:first + BATCH]
            t, k = data.sample_windows(len(index), generator, speed_augmented)
            x, y, mask = data.make_windows(train[index.to(device)], t.to(device), k.to(device))
            x, y = data.to_float(x), data.to_float(y)
            if symmetry_augmented:
                g = torch.randint(0, len(transforms.GROUP), (len(index),), generator=generator).to(device)
                x, y = transforms.apply_per_video(x, g), transforms.apply_per_video(y, g)
            loss = masked_loss(model(x), y, mask)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(index)
            count += len(index)
        torch.cuda.synchronize()
        log["epoch_seconds"].append(time.perf_counter() - start)
        log["train_loss"].append(total / count)

        # Validation is only for monitoring; it is not part of the timed epoch.
        val_prediction = metrics.predict(model, val_x)
        log["val_E5"].append(metrics.per_horizon_mse(val_prediction, val_y.cpu()).mean().item())
        print(f"epoch {epoch + 1:2d}  loss {log['train_loss'][-1]:.5f}  "
              f"val E5 {log['val_E5'][-1]:.5f}  {log['epoch_seconds'][-1]:.1f}s", flush=True)

    RUNS.mkdir(exist_ok=True)
    name = f"{args.model}_s{args.seed}"
    torch.save(model.state_dict(), RUNS / f"{name}.pt")
    (RUNS / f"{name}.json").write_text(json.dumps(log, indent=1))
    print("saved", RUNS / f"{name}.pt")


if __name__ == "__main__":
    main()
