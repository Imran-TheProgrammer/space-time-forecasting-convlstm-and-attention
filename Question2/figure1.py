"""Figure 1: direction and speed of motion in the first 50 training videos (Farneback optical flow)."""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from q2 import data

HERE = Path(__file__).resolve().parent
VIDEOS = 50


def main():
    videos = data.to_float(data.load_videos("train")[:VIDEOS])[:, :, 0].numpy()  # [50, 20, 64, 64]
    as_uint8 = np.round(videos * 255).astype(np.uint8)
    directions, speeds = [], []
    for v in range(VIDEOS):
        for s in range(19):
            flow = cv2.calcOpticalFlowFarneback(as_uint8[v, s], as_uint8[v, s + 1], None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[..., 0], flow[..., 1]
            speed = np.sqrt(dx ** 2 + dy ** 2)
            keep = (videos[v, s] > 0.2) & (videos[v, s + 1] > 0.2) & (speed > 0.5)
            directions.append(np.degrees(np.arctan2(-dy[keep], dx[keep])) % 360)
            speeds.append(speed[keep])
    directions, speeds = np.concatenate(directions), np.concatenate(speeds)
    median = float(np.median(speeds))
    kept = int(len(speeds))

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.arange(0, 361, 20)
    left.hist(directions, bins=bins, weights=np.full(kept, 1 / kept), color="#4c72b0", edgecolor="white")
    left.axhline(1 / 18, color="black", linestyle="--", linewidth=1, label="uniform (1/18)")
    left.set_xticks(bins[::2])
    left.set_xlabel("flow direction θ (degrees; 0° = right, 90° = up)")
    left.set_ylabel("fraction of kept pixels")
    left.set_title("(a) direction")
    left.legend(loc="lower right")

    right.hist(speeds, bins=np.linspace(0.5, np.percentile(speeds, 99.5), 40),
               weights=np.full(kept, 1 / kept), color="#dd8452", edgecolor="white")
    right.axvline(median, color="black", linewidth=1.5, label=f"median = {median:.2f} px/frame")
    right.set_xlabel("flow speed (pixels per frame)")
    right.set_ylabel("fraction of kept pixels")
    right.set_title("(b) speed")
    right.legend()
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "figure1.png", dpi=200)

    counts, _ = np.histogram(directions, bins=bins)
    summary = {"pixels_kept": kept, "median_speed": median,
               "speed_quartiles": [float(q) for q in np.percentile(speeds, [25, 75])],
               "direction_fraction_min": float(counts.min() / kept),
               "direction_fraction_max": float(counts.max() / kept),
               "direction_fractions": (counts / kept).tolist()}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "figure1.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "direction_fractions"}, indent=1))


if __name__ == "__main__":
    main()
