"""Draws the Question 1 architecture diagram (q1/architecture.png)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_name("architecture.png")

SOURCE = "#dbe9f6"
BLOCK = "#fde9c9"
JOIN = "#e3f1dd"
OUTPUT = "#f6d5d5"
EDGE = "#333333"


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor=EDGE, linewidth=1.1))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.45, "\n".join(lines), ha="center", va="top", fontsize=8.6,
            family="monospace", linespacing=1.35)


def arrow(ax, start, end, label=None, offset=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 color=EDGE, linewidth=1.1))
    if label:
        ax.text((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1], label,
                ha="center", va="bottom", fontsize=8, style="italic")


fig, ax = plt.subplots(figsize=(17, 8.6))
ax.set_xlim(0, 17)
ax.set_ylim(0, 8.6)
ax.axis("off")

# Column 1: data sources (all as known at issuance time t0).
ax.text(1.75, 8.45, "1. Data sources (as known at t0)", ha="center", fontsize=11, weight="bold")
sources = [
    ("Water levels S1-S4, T1", ["value  [B,144,5]", "mask   [B,144,5]", "age    [B,144,5]"]),
    ("Rainfall R1, R2", ["mm/10min [B,144,2]", "mask     [B,144,2]", "age      [B,144,2]"]),
    ("Reservoir release", ["actual    [B,144,1]", "plan(past)[B,144,1]", "plan(next)[B,6,1]"]),
    ("Static river map", ["node feat. [8,5]", "(river km, type)", "upstream mask [8,8]"]),
    ("Clock features", ["time of day/year", "[B,150,4]"]),
]
ys = [6.85, 5.25, 3.65, 2.05, 0.6]
hs = [1.4, 1.4, 1.4, 1.4, 1.15]
for (title, lines), y, h in zip(sources, ys, hs):
    box(ax, 0.2, y, 3.1, h, title, lines, SOURCE)

# Column 2: alignment and projection to a common token grid.
box(ax, 4.0, 3.2, 3.2, 3.6, "2. Align + project",
    ["put all series on the", "10-min grid ending at t0", "(k = -143 ... 0)", "",
     "Linear per source type", "-> d = 64 features", "+ node embedding (map)", "+ time embedding", "",
     "stack on node axis:", "Z0 [B,144,8,64]", "future plan tokens:", "P  [B,6,64]"], JOIN)
for y, h in zip(ys[:3], hs[:3]):
    arrow(ax, (3.3, y + h / 2), (4.0, min(max(y + h / 2, 3.4), 6.6)))
arrow(ax, (3.3, ys[3] + hs[3] / 2), (4.0, 3.4))
arrow(ax, (3.3, ys[4] + hs[4] / 2), (4.0, 3.3))

# Column 3: temporal encoder.
box(ax, 7.9, 3.7, 2.7, 2.6, "3. Temporal encoder",
    ["3 causal T blocks", "kernel 3, dilation 1,2,4", "same weights for", "every node",
     "", "Z1 [B,144,8,64]"], BLOCK)
arrow(ax, (7.2, 5.0), (7.9, 5.0))

# Column 4: joint space-time attention.
box(ax, 11.3, 3.2, 2.9, 3.6, "4. Space-time attention",
    ["reshape (time,node)->token", "[B,144,8,64]->[B,1152,64]", "concat P -> [B,1158,64]", "",
     "2 attention blocks,", "4 heads, + feed-forward", "mask: station sees only", "itself, upstream nodes,",
     "T1, rain, reservoir", "", "Z2 [B,1158,64]"], BLOCK)
arrow(ax, (10.6, 5.0), (11.3, 5.0), "flatten")
ax.plot([5.6, 5.6, 12.75], [3.2, 2.85, 2.85], color=EDGE, linewidth=1.1)
arrow(ax, (12.75, 2.85), (12.75, 3.2))
ax.text(9.2, 2.6, "future plan tokens P skip the T blocks", ha="center", fontsize=8, style="italic")

# Column 5: per-station heads and outputs.
box(ax, 14.75, 4.4, 2.05, 2.4, "5. Station heads",
    ["take latest token", "of S1..S4", "-> [B,4,64]", "MLP per station", "-> change [B,4,3]"], BLOCK)
arrow(ax, (14.2, 5.6), (14.75, 5.6))
box(ax, 14.75, 1.7, 2.05, 2.2, "Forecast t0+60",
    ["last level + change", "Y_hat [B,4,3]", "station x", "quantile", "(10, 50, 90 %)"], OUTPUT)
arrow(ax, (15.78, 4.4), (15.78, 3.9))
box(ax, 14.75, 0.2, 2.05, 1.15, "Training loss",
    ["pinball loss vs", "Y [B,4] + mask"], OUTPUT)
arrow(ax, (15.78, 1.7), (15.78, 1.35))

ax.text(8.5, 0.35, "B = batch of issuance times, 144 = 24 h of 10-min steps, 8 nodes = Reservoir, S1, T1, S2, S3, S4, R1, R2,"
        " d = 64 features", ha="center", fontsize=8.6)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"saved {OUT}")
