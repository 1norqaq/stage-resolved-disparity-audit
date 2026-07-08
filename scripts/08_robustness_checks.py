from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from _bootstrap_path import ROOT  # noqa: F401
from stageaudit.io import read_table, write_table
from stageaudit.schema import REQUIRED_SCREEN_COLUMNS, prepare_common, validate_columns


def parse_group(term: str) -> str:
    m = re.search(r"\[T\.(.*)\]", term)
    return m.group(1) if m else term


def _haldane_or(exp_y: pd.Series, ref_y: pd.Series) -> tuple[float, float, float]:
    a = float(exp_y.sum()) + 0.5
    b = float(len(exp_y) - exp_y.sum()) + 0.5
    c = float(ref_y.sum()) + 0.5
    d = float(len(ref_y) - ref_y.sum()) + 0.5
    beta = np.log((a / b) / (c / d))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(np.exp(beta)), float(np.exp(beta - 1.96 * se)), float(np.exp(beta + 1.96 * se))


def fast_screen_table(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows = []
    ref_y = df.loc[df["ethnicity"] == reference, "advanced"]
    for g in sorted([x for x in df["ethnicity"].dropna().unique() if x != reference]):
        exp_y = df.loc[df["ethnicity"] == g, "advanced"]
        if len(exp_y) == 0 or len(ref_y) == 0:
            or_, lo, hi = np.nan, np.nan, np.nan
        else:
            or_, lo, hi = _haldane_or(exp_y, ref_y)
        rows.append({
            "group": g,
            "or": or_,
            "ci_low": lo,
            "ci_high": hi,
            "p_value": np.nan,
            "p_fdr": np.nan,
            "reject_fdr_0_05": False,
            "fit_method": "fast_smoke_rate_or",
        })
    return pd.DataFrame(rows)


def screen_table(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    from stageaudit.stats import add_fdr, coefficient_table, fit_logit

    res = fit_logit(formula, df, cluster="candidate_id")
    tab = coefficient_table(res, "C(ethnicity")
    tab["group"] = tab["term"].map(parse_group)
    return add_fdr(tab)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--reference", default="French")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = prepare_common(read_table(args.input))
    validate_columns(df, REQUIRED_SCREEN_COLUMNS)
    base_cols = ["advanced", "fit_q", "ethnicity", "gender", "candidate_id"]
    base_formula = f"advanced ~ fit_q + C(ethnicity, Treatment(reference='{args.reference}')) + C(gender)"

    rows = []
    for threshold in [0.0, 0.5, 0.7]:
        if "ethnicity_posterior" not in df.columns and threshold > 0:
            continue
        work = df.copy()
        if threshold > 0:
            work = work[work["ethnicity_posterior"] >= threshold]
        work = work.dropna(subset=base_cols).copy()
        tab = fast_screen_table(work, args.reference) if os.environ.get("STAGEAUDIT_FAST", "0") == "1" else screen_table(work, base_formula)
        tab["specification"] = f"posterior_ge_{threshold}"
        tab["n"] = len(work)
        rows.append(tab)
    conf = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    write_table(conf, outdir / "robustness_confidence_thresholds.csv")

    if os.environ.get("STAGEAUDIT_FAST", "0") == "1":
        work = df.dropna(subset=base_cols).copy()
        tab = fast_screen_table(work, args.reference)
        tab["specification"] = "sorting_controls_fast_smoke_placeholder"
        tab["n"] = len(work)
        write_table(tab, outdir / "robustness_sorting_controls.csv")
        print(f"wrote robustness outputs to {outdir} (fast synthetic smoke approximation)")
        return

    controls = []
    if "job_family" in df.columns:
        controls.append("C(job_family)")
    if "job_category" in df.columns:
        controls.append("C(job_category)")
    if "application_region" in df.columns:
        controls.append("C(application_region)")
    if "application_month" in df.columns:
        controls.append("C(application_month)")
    if controls:
        work = df.dropna(subset=base_cols).copy()
        formula = base_formula + " + " + " + ".join(controls)
        tab = screen_table(work, formula)
        tab["specification"] = "sorting_controls"
        tab["n"] = len(work)
        write_table(tab, outdir / "robustness_sorting_controls.csv")

    print(f"wrote robustness outputs to {outdir}")


if __name__ == "__main__":
    main()
