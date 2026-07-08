from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd
from _bootstrap_path import ROOT  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing result tables")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    indir = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("STAGEAUDIT_FAST", "0") == "1":
        (outdir / "SMOKE_FIGURES_SKIPPED.txt").write_text(
            "Synthetic smoke mode completed table generation. Full figures are generated in non-fast mode.\n"
        )
        print(f"wrote smoke figure placeholder to {outdir}")
        return

    from stageaudit.plots import forest_plot

    screen = indir / "screen_gate.csv"
    if screen.exists():
        df = pd.read_csv(screen)
        forest_plot(df, "group", "or", "ci_low", "ci_high", outdir / "screen_gate_forest.png", "CV screen odds ratios")

    hire = indir / "hire_gate_bootstrap.csv"
    if hire.exists():
        df = pd.read_csv(hire).dropna(subset=["ci_low", "ci_high"])
        if len(df):
            forest_plot(df, "group", "or", "ci_low", "ci_high", outdir / "hire_gate_forest.png", "Hire-gate odds ratios")

    print(f"wrote figures to {outdir}")


if __name__ == "__main__":
    main()
