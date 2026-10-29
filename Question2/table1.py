"""Table 1: checks that the transformation and window functions match the handout."""
import json
from pathlib import Path

import torch

from q2 import data, latex, transforms

RESULTS = Path(__file__).resolve().parent / "results"


def exact(a, b):
    return {"max_abs_diff": (a - b).abs().max().item(), "equal": bool(torch.equal(a, b))}


def bright_pixel(video):
    return tuple(int(i) for i in (video == video.max()).nonzero()[0][[0, 2, 3]])


def frames_used(k):
    """Run the window function with t = 1 on a video whose frame f is filled with f."""
    video = torch.arange(1, 21, dtype=torch.float32)[:, None, None, None].expand(20, 1, 64, 64)
    x, y, mask = data.fixed_windows(video[None], k=k)
    inputs = [int(v) for v in x[0, :, 0, 0, 0]]
    targets = [int(v) for v, m in zip(y[0, :, 0, 0, 0], mask[0]) if m]
    return inputs, targets


def main():
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1

    r4 = x
    for _ in range(4):
        r4 = transforms.rotate(r4)
    checks = {
        "a": exact(transforms.mirror(transforms.mirror(x)), x),
        "b": exact(r4, x),
        "c": exact(transforms.mirror(transforms.rotate(transforms.mirror(x))), transforms.rotate(x, -1)),
        "d": bright_pixel(transforms.mirror(p)),
        "e": bright_pixel(transforms.rotate(p)),
        "f": frames_used(1), "g": frames_used(2), "h": frames_used(3),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "table1.json").write_text(json.dumps(checks, indent=1))

    def diff(c):
        return f"{latex.number(c['max_abs_diff'])} (\\texttt{{torch.equal}}: {c['equal']})"

    def frames(c):
        return f"{', '.join(map(str, c[0]))} / {', '.join(map(str, c[1]))}"

    rows = [
        ["(a)", "$\\max|M(M(x)) - x|$", "0", diff(checks["a"])],
        ["(b)", "$\\max|R^4(x) - x|$", "0", diff(checks["b"])],
        ["(c)", "$\\max|MRM(x) - R^{-1}(x)|$", "0", diff(checks["c"])],
        ["(d)", "bright pixel of $M(p)$ (frame, row, column)", "(7, 10, 43)", str(checks["d"])],
        ["(e)", "bright pixel of $R(p)$ (frame, row, column)", "(7, 43, 10)", str(checks["e"])],
        "midrule",
        ["(f)", "1$\\times$ input / target frames", "1--5 / 6--10", frames(checks["f"])],
        ["(g)", "2$\\times$ input / target frames", "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19", frames(checks["g"])],
        ["(h)", "3$\\times$ input / target frames", "1, 4, 7, 10, 13 / 16, 19", frames(checks["h"])],
    ]
    latex.write("table1", ["", "Check", "Expected", "Yours"], rows, "llll")
    for key, value in checks.items():
        print(key, value)


if __name__ == "__main__":
    main()
