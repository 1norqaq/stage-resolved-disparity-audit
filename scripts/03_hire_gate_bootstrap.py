from __future__ import annotations

import argparse
import re
import numpy as np
import pandas as pd
from _bootstrap_path import ROOT  # noqa: F401
from stageaudit.io import read_table, write_table
from stageaudit.schema import REQUIRED_HIRE_COLUMNS, prepare_common, validate_columns
from stageaudit.stats import bootstrap_cluster, coefficient_table, fit_logit_ridge


def parse_group(term: str) -> str:
    m = re.search(r"\[T\.(.*)\]", term)
    return m.group(1) if m else term


def fit_once(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Fast finite hire-gate ORs for sparse downstream/bootstrap samples."""
    formula = f"hired ~ fit_q + C(ethnicity, Treatment(reference='{reference}')) + C(gender)"
    result = fit_logit_ridge(formula, df)
    tab = coefficient_table(result, "C(ethnicity")
    tab["group"] = tab["term"].map(parse_group)
    return tab[["group", "or", "fit_method"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", default="French")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    df = prepare_common(read_table(args.input))
    validate_columns(df, REQUIRED_HIRE_COLUMNS)
    df = df[df["advanced"] == 1].dropna(subset=["hired", "fit_q", "ethnicity", "gender", "candidate_id"]).copy()
    if len(df) == 0 or df["hired"].nunique() < 2:
        out = pd.DataFrame(columns=["group", "or", "ci_low", "ci_high", "successful_bootstraps", "requested_bootstraps", "fit_method"])
        write_table(out, args.out)
        print(f"wrote {args.out} (no identifiable hire model)")
        return

    point = fit_once(df, args.reference).rename(columns={"or": "or"})

    boot_rows = []
    failed = 0
    for b, sample in enumerate(bootstrap_cluster(df, "candidate_id", args.bootstrap, args.seed)):
        try:
            if sample["hired"].nunique() < 2:
                failed += 1
                continue
            tmp = fit_once(sample, args.reference)
            tmp["bootstrap"] = b
            boot_rows.append(tmp)
        except Exception:
            failed += 1
            continue
    boot = pd.concat(boot_rows, ignore_index=True) if boot_rows else pd.DataFrame(columns=["group", "or", "bootstrap"])
    if len(boot):
        intervals = boot.groupby("group")["or"].quantile([0.025, 0.975]).unstack().reset_index()
        intervals.columns = ["group", "ci_low", "ci_high"]
        counts = boot.groupby("group")["bootstrap"].nunique().reset_index(name="successful_bootstraps")
        intervals = intervals.merge(counts, on="group", how="left")
    else:
        intervals = pd.DataFrame(columns=["group", "ci_low", "ci_high", "successful_bootstraps"])
    out = point.merge(intervals, on="group", how="left")
    out["requested_bootstraps"] = args.bootstrap
    out["failed_bootstraps_total"] = failed
    write_table(out, args.out)
    print(f"wrote {args.out}; successful bootstrap fits: {len(boot):,} rows, failed resamples: {failed}")


if __name__ == "__main__":
    main()
