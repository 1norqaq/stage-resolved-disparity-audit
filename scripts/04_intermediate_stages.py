from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from _bootstrap_path import ROOT  # noqa: F401
from stageaudit.io import read_table, write_table
from stageaudit.schema import prepare_common

MILESTONES = ["prequalified", "interview_shortlist", "offered"]
DISADVANTAGED = ["Muslim", "African", "Indian-subcontinent"]
REFERENCE = "French"


def _haldane_log_or(y_exposed: pd.Series, y_unexposed: pd.Series) -> tuple[float, float]:
    a = float(y_exposed.sum()) + 0.5
    b = float(len(y_exposed) - y_exposed.sum()) + 0.5
    c = float(y_unexposed.sum()) + 0.5
    d = float(len(y_unexposed) - y_unexposed.sum()) + 0.5
    beta = np.log((a / b) / (c / d))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(beta), float(se)


def _wald_or(beta: float, se: float) -> tuple[float, float, float]:
    return float(np.exp(beta)), float(np.exp(beta - 1.96 * se)), float(np.exp(beta + 1.96 * se))


def fit_fast_smoke(df: pd.DataFrame, outcome: str) -> dict:
    work = df[df["ethnicity"].isin([REFERENCE] + DISADVANTAGED)].dropna(subset=[outcome, "ethnicity"]).copy()
    exp = work[work["ethnicity"].isin(DISADVANTAGED)][outcome]
    ref = work[work["ethnicity"] == REFERENCE][outcome]
    if len(exp) == 0 or len(ref) == 0:
        beta, se = np.nan, np.nan
        or_, lo, hi = np.nan, np.nan, np.nan
    else:
        beta, se = _haldane_log_or(exp, ref)
        or_, lo, hi = _wald_or(beta, se)
    return {
        "milestone": outcome,
        "estimate": beta,
        "std_error": se,
        "or": or_,
        "ci_low": lo,
        "ci_high": hi,
        "n": len(work),
        "events": int(work[outcome].sum()) if outcome in work else 0,
        "fit_method": "fast_smoke_rate_or",
    }


def fit_penalized(df: pd.DataFrame, outcome: str) -> dict:
    from stageaudit.stats import fit_logit_ridge, wald_or

    work = df[df["ethnicity"].isin([REFERENCE] + DISADVANTAGED)].dropna(subset=[outcome, "fit_q", "gender", "ethnicity"]).copy()
    work["disadvantaged"] = work["ethnicity"].isin(DISADVANTAGED).astype(int)
    if len(work) == 0 or work[outcome].nunique() < 2 or work["disadvantaged"].nunique() < 2:
        return {
            "milestone": outcome,
            "estimate": np.nan,
            "std_error": np.nan,
            "or": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": len(work),
            "events": int(work[outcome].sum()) if outcome in work else 0,
            "fit_method": "not_identifiable",
        }
    formula = f"{outcome} ~ disadvantaged + fit_q + C(gender)"
    result = fit_logit_ridge(formula, work)
    beta = float(result.params["disadvantaged"])
    se = float(result.bse["disadvantaged"])
    or_, lo, hi = wald_or(beta, se)
    return {
        "milestone": outcome,
        "estimate": beta,
        "std_error": se,
        "or": or_,
        "ci_low": lo,
        "ci_high": hi,
        "n": len(work),
        "events": int(work[outcome].sum()),
        "fit_method": getattr(result, "method", "ridge-logit"),
    }


def empirical_bayes(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    ok = rows["estimate"].notna() & rows["std_error"].notna() & (rows["std_error"] > 0)
    if ok.sum() == 0:
        return rows
    y = rows.loc[ok, "estimate"].to_numpy()
    v = rows.loc[ok, "std_error"].to_numpy() ** 2
    mean_fe = np.sum(y / v) / np.sum(1 / v)
    q = np.sum((y - mean_fe) ** 2 / v)
    denom = max(1e-12, np.sum(1 / v) - np.sum((1 / v) ** 2) / np.sum(1 / v))
    tau2 = max(0.0, (q - (len(y) - 1)) / denom)
    shrunk = []
    for yi, vi in zip(y, v):
        weight = tau2 / (tau2 + vi) if tau2 + vi > 0 else 0.0
        shrunk.append(weight * yi + (1 - weight) * mean_fe)
    rows["eb_estimate"] = np.nan
    rows.loc[ok, "eb_estimate"] = shrunk
    rows["eb_or"] = np.exp(rows["eb_estimate"])
    rows["tau2"] = tau2
    common_se = np.sqrt(1 / np.sum(1 / (v + tau2)))
    common_est = np.sum(y / (v + tau2)) / np.sum(1 / (v + tau2))
    rows["common_estimate"] = common_est
    rows["common_or"] = np.exp(common_est)
    rows["common_ci_low"] = np.exp(common_est - 1.96 * common_se)
    rows["common_ci_high"] = np.exp(common_est + 1.96 * common_se)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = prepare_common(read_table(args.input))
    present = [m for m in MILESTONES if m in df.columns]
    fitter = fit_fast_smoke if os.environ.get("STAGEAUDIT_FAST", "0") == "1" else fit_penalized
    rows = [fitter(df, m) for m in present]
    out = empirical_bayes(pd.DataFrame(rows))
    write_table(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
