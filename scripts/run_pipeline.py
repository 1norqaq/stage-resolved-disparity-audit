from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, fast: bool) -> None:
    env = os.environ.copy()
    if fast:
        env["STAGEAUDIT_FAST"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stage-audit pipeline end to end.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--fast", action="store_true", help="Use fast/safe synthetic-data code paths for smoke tests only.")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    py = args.python
    inp = args.input
    commands = [
        [py, "scripts/00_validate_inputs.py", "--input", inp],
        [py, "scripts/01_screen_gate_disparate_impact.py", "--input", inp, "--out", "results/tables/screen_gate.csv"],
        [py, "scripts/02_selection_rate_ratios.py", "--input", inp, "--out", "results/tables/selection_rate_ratios.csv"],
        [py, "scripts/03_hire_gate_bootstrap.py", "--input", inp, "--out", "results/tables/hire_gate_bootstrap.csv", "--bootstrap", str(args.bootstrap)],
        [py, "scripts/04_intermediate_stages.py", "--input", inp, "--out", "results/tables/intermediate_stages.csv"],
        [py, "scripts/05_mediation_evalues.py", "--input", inp, "--out", "results/tables/mediation_evalues.csv"],
        [py, "scripts/06_representation_probe.py", "--input", inp, "--out", "results/tables/representation_probe.csv"],
        [py, "scripts/07_content_leakage.py", "--input", inp, "--out", "results/tables/content_leakage.csv"],
        [py, "scripts/08_robustness_checks.py", "--input", inp, "--outdir", "results/tables"],
        [py, "scripts/09_make_figures.py", "--input", "results/tables", "--outdir", "results/figures"],
    ]
    for cmd in commands:
        run(cmd, fast=args.fast)


if __name__ == "__main__":
    main()
