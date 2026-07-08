from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def forest_plot(df: pd.DataFrame, label_col: str, or_col: str, low_col: str, high_col: str, out: str | Path, title: str) -> None:
    data = df.copy().sort_values(or_col)
    y = range(len(data))
    plt.figure(figsize=(7, max(3, 0.35 * len(data))))
    lower = (data[or_col] - data[low_col]).clip(lower=0)
    upper = (data[high_col] - data[or_col]).clip(lower=0)
    plt.errorbar(data[or_col], y, xerr=[lower, upper], fmt="o", capsize=3)
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.axvline(0.8, linestyle=":", linewidth=1)
    plt.yticks(list(y), data[label_col])
    plt.xlabel("Odds ratio")
    plt.title(title)
    plt.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()
