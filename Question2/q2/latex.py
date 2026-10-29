"""Number formatting and table output for the LaTeX report."""
import math
from pathlib import Path

TABLES = Path(__file__).resolve().parents[2] / "report" / "tables"


def number(value):
    """3 significant figures; scientific notation below 1e-3."""
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        exponent = math.floor(math.log10(abs(value)))
        mantissa = value / 10 ** exponent
        if round(abs(mantissa), 2) >= 10:
            mantissa, exponent = mantissa / 10, exponent + 1
        return f"${mantissa:.2f}\\times10^{{{exponent}}}$"
    rounded = float(f"{value:.3g}")
    digits = max(3 - 1 - math.floor(math.log10(abs(rounded))), 0)
    return f"{rounded:,.{digits}f}"


def seeds(values):
    """mean (seed 0, seed 1)."""
    mean = sum(values) / len(values)
    return f"{number(mean)} ({', '.join(number(v) for v in values)})"


def integer(value):
    return f"{int(value):,}"


def write(name, header, rows, column_spec):
    """Write report/tables/<name>.tex as a booktabs tabular."""
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [f"\\begin{{tabular}}{{{column_spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for row in rows:
        lines.append("\\midrule" if row == "midrule" else " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")
