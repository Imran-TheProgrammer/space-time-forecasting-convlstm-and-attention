"""Prediction helpers and the error measures: MSE_h, E(5), E(2), C_T and E_T."""
import time

import torch

from . import data, transforms

BATCH = 32


@torch.no_grad()
def predict(model, x, batch=BATCH):
    """Run model on x [N, 5, 1, 64, 64] in batches; returns predictions on the CPU."""
    model.eval()
    parameters = list(model.parameters())
    device = parameters[0].device if parameters else x.device
    out = [model(x[i:i + batch].to(device)).cpu() for i in range(0, len(x), batch)]
    return torch.cat(out)


def per_horizon_mse(prediction, target):
    """Mean squared error over the 64 x 64 pixels of each horizon: [N, 5]."""
    return ((prediction - target) ** 2).mean(dim=(2, 3, 4))


def per_video(prediction, target, mask=None):
    """Squared error averaged over pixels and (available) horizons of each video: [N]."""
    errors = per_horizon_mse(prediction, target)
    if mask is None:
        return errors.mean(dim=1)
    return (errors * mask).sum(dim=1) / mask.sum(dim=1)


def evaluate(model, test_videos, original_only=False):
    """Every test-set number used in Tables 2-6, kept per video so seeds can be combined.

    test_videos: uint8 [1000, 20, 1, 64, 64]. All windows start at t = 1.
    original_only: only the original 1x clips (Table 2), with no transformed or faster clips.
    """
    results = {}
    for k in ((1,) if original_only else (1, 2, 3)):
        x, y, mask = data.fixed_windows(test_videos, k=k)
        x, y = data.to_float(x), data.to_float(y)
        prediction = predict(model, x)
        mse = per_horizon_mse(prediction, y)  # unavailable horizons are never used
        results[f"mse_{k}x"] = mse
        results[f"E2_{k}x"] = mse[:, :2].mean(dim=1)
        if mask.cpu().min() == 1:  # E(5) only where all five targets exist (1x and 2x)
            results[f"E5_{k}x"] = mse.mean(dim=1)
        if k == 1 and not original_only:
            # Consistency gap C_T = MSE(f(T(X)), T(f(X))) and transformed error E_T = MSE(f(T(X)), T(Y)).
            for name, transform in (("M", transforms.mirror), ("R", transforms.rotate)):
                transformed = predict(model, transform(x))
                results[f"C_{name}"] = per_video(transformed, transform(prediction))
                results[f"E_{name}"] = per_video(transformed, transform(y))
    return results


@torch.no_grad()
def inference_seconds(model, test_videos, device):
    """Wall-clock time to predict all original (1x, t = 1) test videos at batch size 32.

    Each batch is moved to the device before its timer starts. One untimed warm-up
    pass runs first so that one-off GPU set-up is not counted.
    """
    model.eval()
    x, _, _ = data.fixed_windows(test_videos, k=1)
    batches = [data.to_float(x[i:i + BATCH]).to(device) for i in range(0, len(x), BATCH)]
    for b in batches:
        model(b)
    total = 0.0
    for b in batches:
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(b)
        torch.cuda.synchronize()
        total += time.perf_counter() - start
    return total
