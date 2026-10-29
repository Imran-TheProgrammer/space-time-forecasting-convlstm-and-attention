"""Moving MNIST loading, the split by video index, and speed windows."""
from pathlib import Path

import torch
from torchvision.datasets import MovingMNIST

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

SPLITS = {"train": (0, 8000), "val": (8000, 9000), "test": (9000, 10000)}
FRAMES = 20
CONTEXT = 5
HORIZONS = 5

# Largest allowed start frame t (frames numbered 1-20) for each speed k.
# Normal-speed training (models A, B): t in 1..11 at k = 1.
# Speed-augmented training (models C, D): all 5 inputs and at least 2 targets exist.
NORMAL_MAX_START = 11
AUGMENTED_MAX_START = {k: FRAMES - 6 * k for k in (1, 2, 3)}  # {1: 14, 2: 8, 3: 2}


def load_videos(split=None):
    """All videos as uint8 [N, 20, 1, 64, 64]; split by index, never by frame."""
    videos = MovingMNIST(root=str(DATA_DIR), split=None, download=True).data
    if split is None:
        return videos
    start, stop = SPLITS[split]
    return videos[start:stop]


def to_float(frames):
    """uint8 frames -> float32 in [0, 1]."""
    return frames.float() / 255.0


def window_indices(t, k):
    """Frame indices (0-based) and target mask for windows starting at frame t with speed k.

    t, k: integer tensors [B] (t is 1-based, as in the handout).
    Returns input indices [B, 5], target indices [B, 5] (clamped so they can be
    gathered) and a mask [B, 5] that is 1 where the target frame exists (<= 20).
    """
    steps = torch.arange(CONTEXT + HORIZONS, device=t.device)
    frames = t[:, None] + steps[None, :] * k[:, None]  # 1-based frame numbers
    mask = (frames[:, CONTEXT:] <= FRAMES).float()
    index = frames.clamp(max=FRAMES) - 1
    return index[:, :CONTEXT], index[:, CONTEXT:], mask


def make_windows(videos, t, k):
    """Cut windows X [B, 5, 1, H, W] and Y [B, 5, 1, H, W] from videos [B, 20, 1, H, W].

    Target frames past frame 20 are filled with zeros and marked 0 in the mask.
    """
    input_index, target_index, mask = window_indices(t, k)
    rows = torch.arange(videos.shape[0], device=videos.device)[:, None]
    x = videos[rows, input_index]
    y = videos[rows, target_index] * mask[:, :, None, None, None].to(videos.dtype)
    return x, y, mask


def fixed_windows(videos, k, t=1):
    """Windows with the same start t and speed k for every video."""
    n = videos.shape[0]
    t = torch.full((n,), t, dtype=torch.long, device=videos.device)
    k = torch.full((n,), k, dtype=torch.long, device=videos.device)
    return make_windows(videos, t, k)


def sample_windows(n, generator, speed_augmented):
    """Draw one start t and speed k per video, following the training rules."""
    if speed_augmented:
        k = torch.randint(1, 4, (n,), generator=generator)
        max_start = torch.tensor([0] + [AUGMENTED_MAX_START[s] for s in (1, 2, 3)])[k]
        t = 1 + (torch.rand(n, generator=generator) * max_start).long()
    else:
        k = torch.ones(n, dtype=torch.long)
        t = torch.randint(1, NORMAL_MAX_START + 1, (n,), generator=generator)
    return t, k
