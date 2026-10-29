"""Tables 2-6 as LaTeX, from results/eval/*.pt and runs/*.json.

Usage: python make_tables.py table2 [table3 table4 table5 table6 --final symDB]
Learned models are shown as mean (seed 0, seed 1). Measures are rows, models are columns.
"""
import argparse
import json
from pathlib import Path

import torch

from q2 import latex

HERE = Path(__file__).resolve().parent
EVAL = HERE / "results" / "eval"
SEEDS = (0, 1)
LABELS = {"Blank": "Blank", "CopyLast": "Copy-last", "symB": "Sym.\\ B", "symD": "Sym.\\ D", "symDB": "Sym.\\ D+B"}


def runs(name):
    if name in ("Blank", "CopyLast"):
        return [torch.load(EVAL / f"{name}.pt")]
    return [torch.load(EVAL / f"{name}_s{s}.pt") for s in SEEDS]


def cell(name, measure):
    """One table entry: a single value for Blank/Copy-last, mean (seed 0, seed 1) otherwise."""
    values = []
    for result in runs(name):
        if measure.startswith("MSE"):
            values.append(result["mse_1x"][:, int(measure[3:]) - 1].mean().item())
        elif measure == "parameters":
            return latex.integer(result["parameters"])
        elif measure == "inference":
            values.append(result["inference_seconds"])
        else:
            values.append(result[measure].mean().item())
    return latex.number(values[0]) if len(values) == 1 else latex.seeds(values)


def train_time(name):
    if name in ("Blank", "CopyLast"):
        return "--"
    base = name[3:] if name.startswith("sym") else name
    times = []
    for s in SEEDS:
        log = json.loads((HERE / "runs" / f"{base}_s{s}.json").read_text())
        times.append(sum(log["epoch_seconds"]) / len(log["epoch_seconds"]))
    return latex.seeds(times)


def table(file, names, measures):
    header = ["Measure"] + [LABELS.get(n, f"Model {n}") for n in names]
    rows = []
    for label, measure in measures:
        if measure == "train":
            rows.append([label] + [train_time(n) for n in names])
        else:
            rows.append([label] + [cell(n, measure) for n in names])
    latex.write(file, header, rows, "l" + "c" * len(names))


FULL = [("$E^{(5)}$ at 1$\\times$", "E5_1x"),
        ("$C_M$", "C_M"), ("$E_M$", "E_M"), ("$C_R$", "C_R"), ("$E_R$", "E_R"),
        ("$E^{(2)}$ at 1$\\times$", "E2_1x"), ("$E^{(2)}$ at 2$\\times$", "E2_2x"), ("$E^{(2)}$ at 3$\\times$", "E2_3x"),
        ("Trainable parameters", "parameters"),
        ("Training time per epoch (s)", "train"),
        ("Inference time, 1,000 videos (s)", "inference")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tables", nargs="+")
    parser.add_argument("--final", default=None, help="final model if it is not symD, e.g. symDB")
    args = parser.parse_args()
    for t in args.tables:
        if t == "table2":
            measures = [(f"MSE$_{h}$", f"MSE{h}") for h in range(1, 6)]
            measures += [("$E^{(5)}$", "E5_1x"), ("Trainable parameters", "parameters")]
            table("table2", ["A", "Blank", "CopyLast"], measures)
        elif t == "table3":
            table("table3", ["A", "Blank", "CopyLast"], [("$E^{(5)}$", "E5_1x"), ("$C_M$", "C_M"), ("$E_M$", "E_M"),
                                                         ("$C_R$", "C_R"), ("$E_R$", "E_R")])
        elif t == "table4":
            table("table4", ["A", "CopyLast"], [("$E^{(2)}$ at 1$\\times$", "E2_1x"), ("$E^{(2)}$ at 2$\\times$", "E2_2x"),
                                                ("$E^{(2)}$ at 3$\\times$", "E2_3x"), ("$E^{(5)}$ at 1$\\times$", "E5_1x"),
                                                ("$E^{(5)}$ at 2$\\times$", "E5_2x")])
        elif t == "table5":
            table("table5", ["A", "B", "C", "D"], FULL)
        elif t == "table6":
            names = ["symB", "symD"] + ([args.final] if args.final and args.final != "symD" else [])
            table("table6", names, FULL)
        print("wrote", t)


if __name__ == "__main__":
    main()
