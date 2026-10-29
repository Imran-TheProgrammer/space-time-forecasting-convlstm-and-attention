"""Figures 2-6 and the GIFs.

Usage: python figures.py fig2 [fig3 fig4 fig5] [fig6 --final symD]
All figures use seed-0 models. Chosen test videos are stored in results/figure_videos.json
so that Figure 6 reuses exactly the videos of Figures 3 and 5.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from q2 import data, models, transforms

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
GIFS = HERE / "gifs"
CHOICES = HERE / "results" / "figure_videos.json"
TEST_START = data.SPLITS["test"][0]
TRANSFORMS = {"M": transforms.mirror, "R": transforms.rotate}


def load_model(name, device):
    symmetrised = name.startswith("sym")
    base = name[3:] if symmetrised else name
    model = models.build(base)
    model.load_state_dict(torch.load(HERE / "runs" / f"{base}_s0.pt", map_location="cpu"))
    model = model.to(device).eval()
    return models.Symmetrised(model) if symmetrised else model


@torch.no_grad()
def run(model, x, device):
    return model(x.to(device)).cpu()


def digit_blobs(frame):
    """Number of separate bright blobs (area >= 20 px) in a [1, 64, 64] float frame."""
    mask = (frame[0].numpy() > 0.3).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return int((stats[1:, cv2.CC_STAT_AREA] >= 20).sum())


def overlapping(frames):
    """True if the two digits merge into one blob in any of the frames."""
    return any(digit_blobs(f) < 2 for f in frames)


def separate(frames):
    return all(digit_blobs(f) == 2 for f in frames)


def load_choices():
    return json.loads(CHOICES.read_text()) if CHOICES.exists() else {}


def save_choices(choices):
    CHOICES.parent.mkdir(exist_ok=True)
    CHOICES.write_text(json.dumps(choices, indent=1))


def grid(rows, row_labels, column_labels, title, path, split_after=5):
    """rows: list of lists of [1, 64, 64] frames (None = empty cell)."""
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.05 * n_cols + 1.3, 1.08 * n_rows + 0.5), squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            frame = row[c] if c < len(row) else None
            if frame is None:
                ax.axis("off")
                continue
            ax.imshow(frame[0].clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
            if c == split_after:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#d62728")
            if r == 0 and column_labels:
                ax.set_title(column_labels[c], fontsize=8)
        axes[r, 0].set_ylabel(row_labels[r], fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def consistency_panel(model, video, transform_name, device, path_png, path_gif, title):
    """4 rows x 10 frames: X|f(X), T(X)|T(Y), T(X)|f(T(X)), T(X)|T(f(X)); plus a GIF over horizons."""
    transform = TRANSFORMS[transform_name]
    x, y, _ = data.fixed_windows(video[None], k=1)
    x, y = data.to_float(x), data.to_float(y)
    fx = run(model, x, device)
    tx = transform(x)
    ftx = run(model, tx, device)
    rows = [(x[0], fx[0]), (tx[0], transform(y)[0]), (tx[0], ftx[0]), (tx[0], transform(fx)[0])]
    t = transform_name
    labels = ["(1) X | f(X)", f"(2) {t}(X) | {t}(Y)", f"(3) {t}(X) | f({t}(X))", f"(4) {t}(X) | {t}(f(X))"]
    columns = [f"in {i}" for i in range(1, 6)] + [f"h={h}" for h in range(1, 6)]
    grid([list(a) + list(b) for a, b in rows], labels, columns, title, path_png)

    images = []
    for h in range(5):
        fig, axes = plt.subplots(4, 2, figsize=(3.6, 6.4))
        for r, (inputs, future) in enumerate(rows):
            for c, (frame, name) in enumerate(((inputs[-1], "last input"), (future[h], f"horizon {h + 1}"))):
                ax = axes[r, c]
                ax.imshow(frame[0].clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(name, fontsize=9)
            axes[r, 0].set_ylabel(labels[r], fontsize=7)
        fig.suptitle(title, fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    images[0].save(path_gif, save_all=True, append_images=images[1:], duration=700, loop=0)


def figure2(test, device):
    model = load_model("A", device)
    x, y, _ = data.fixed_windows(test, k=1)
    x, y = data.to_float(x), data.to_float(y)
    clean = [i for i in range(len(test)) if separate(torch.cat([x[i], y[i]]))]
    overlap = [i for i in range(len(test)) if overlapping(y[i])]
    chosen = clean[:2] + overlap[:1]
    rows, labels = [], []
    for i in chosen:
        prediction = run(model, x[i:i + 1], device)[0]
        rows += [list(x[i]) + list(prediction), [None] * 5 + list(y[i])]
        tag = " (overlap)" if i in overlap else ""
        labels += [f"video {TEST_START + i}{tag}\ninput | prediction", "ground truth"]
    columns = [f"frame {f}" for f in range(1, 11)]
    grid(rows, labels, columns, "Figure 2: model A (seed 0), input frames 1-5, predicted vs true frames 6-10",
         FIGURES / "figure2.png")
    choices = load_choices()
    choices["figure2"] = [TEST_START + i for i in chosen]
    save_choices(choices)
    print("figure 2 videos:", choices["figure2"])


def figure3(test, device, name="A", label="figure3"):
    choices = load_choices()
    if "figure3" not in choices:
        results = torch.load(HERE / "results" / "eval" / "A_s0.pt")
        choices["figure3"] = {t: [TEST_START + int(i) for i in results[f"C_{t}"].argsort(descending=True)[:2]]
                              for t in TRANSFORMS}
        save_choices(choices)
    model = load_model(name, device)
    shown = "model A" if name == "A" else "final model"
    paths = []
    for t, videos in choices["figure3"].items():
        for rank, video_id in enumerate(videos, 1):
            stem = f"{label}_{t}{rank}_video{video_id}"
            title = f"{shown}, T = {t}, rank {rank} by C_{t}, test video {video_id}"
            consistency_panel(model, test[video_id - TEST_START], t, device,
                              FIGURES / f"{stem}.png", GIFS / f"{stem}.gif", title)
            paths.append(f"Question2/gifs/{stem}.gif")
    print("\n".join(paths))


def figure4():
    results = torch.load(HERE / "results" / "eval" / "A_s0.pt")
    marked = load_choices()["figure3"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, t in zip(axes, TRANSFORMS):
        values = results[f"C_{t}"].numpy()
        ax.hist(values, bins=50, color="#4c72b0", edgecolor="white")
        for rank, video_id in enumerate(marked[t], 1):
            v = values[video_id - TEST_START]
            ax.axvline(v, color="#d62728", linestyle="--", linewidth=1.2)
            ax.annotate(f"video {video_id}", (v, ax.get_ylim()[1] * (0.9 - 0.12 * rank)), fontsize=8,
                        ha="right", color="#d62728")
        ax.set_xlabel(f"per-video $C_{t}$")
        ax.set_ylabel("number of test videos")
        ax.set_title(f"({'a' if t == 'M' else 'b'}) $C_{t}$, model A seed 0, mean = {values.mean():.4f}")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4.png", dpi=200)
    plt.close(fig)


def figure5(test, device, name="A", label="figure5"):
    choices = load_choices()
    if "figure5" not in choices:
        for i in range(len(test)):
            frames = []
            for k in (1, 2, 3):
                _, y, _ = data.fixed_windows(test[i:i + 1], k=k)
                frames += list(data.to_float(y)[0, :2])
            if separate(frames):
                choices["figure5"] = TEST_START + i
                break
        save_choices(choices)
    video_id = choices["figure5"]
    model = load_model(name, device)
    rows, labels = [], []
    for k in (1, 2, 3):
        x, y, _ = data.fixed_windows(test[video_id - TEST_START][None], k=k)
        x, y = data.to_float(x), data.to_float(y)
        prediction = run(model, x, device)
        rows.append(list(x[0]) + list(prediction[0, :2]) + list(y[0, :2]))
        input_frames = ", ".join(str(1 + j * k) for j in range(5))
        labels.append(f"{k}x (inputs {input_frames};\ntargets {1 + 5 * k}, {1 + 6 * k})")
    columns = [f"in {j}" for j in range(1, 6)] + ["pred h=1", "pred h=2", "true h=1", "true h=2"]
    shown = "model A (seed 0)" if name == "A" else "final model (seed 0)"
    grid(rows, labels, columns, f"{shown}, test video {video_id}, at 1x, 2x and 3x", FIGURES / f"{label}.png")
    print("figure 5 video:", video_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("figures", nargs="+")
    parser.add_argument("--final", default="symD")
    args = parser.parse_args()
    FIGURES.mkdir(exist_ok=True)
    GIFS.mkdir(exist_ok=True)
    device = torch.device("cuda")
    test = data.load_videos("test")
    for f in args.figures:
        if f == "fig2":
            figure2(test, device)
        elif f == "fig3":
            figure3(test, device)
        elif f == "fig4":
            figure4()
        elif f == "fig5":
            figure5(test, device)
        elif f == "fig6":
            figure3(test, device, name=args.final, label="figure6a")
            figure5(test, device, name=args.final, label="figure6b")


if __name__ == "__main__":
    main()
