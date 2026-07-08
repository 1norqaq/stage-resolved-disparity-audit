from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from _bootstrap_path import ROOT  # noqa: F401
from stageaudit.io import read_table, write_table
from stageaudit.schema import REQUIRED_SCREEN_COLUMNS, prepare_common, validate_columns

DISADVANTAGED = ["Muslim", "African", "Indian-subcontinent"]


def evalue_for_rr(rr: float) -> float:
    rr = float(rr)
    if rr <= 0:
        return float("nan")
    if rr < 1:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def _haldane_or(y_exposed: pd.Series, y_unexposed: pd.Series) -> float:
    a = float(y_exposed.sum()) + 0.5
    b = float(len(y_exposed) - y_exposed.sum()) + 0.5
    c = float(y_unexposed.sum()) + 0.5
    d = float(len(y_unexposed) - y_unexposed.sum()) + 0.5
    return (a / b) / (c / d)


def fast_smoke_table(work: pd.DataFrame) -> pd.DataFrame:
    """Instant synthetic-data approximation used only for make smoke/demo.

    The full governed-data run uses the model-based decomposition below.  This
    branch exists so reviewers can verify file I/O and downstream table schemas
    even on tiny synthetic data where mediation logits can be ill-conditioned.
    """
    rows = []
    work = work.copy()
    work["disadvantaged"] = work["ethnicity"].isin(DISADVANTAGED).astype(int)
    ref = work[work["disadvantaged"] == 0]["advanced"]
    exp = work[work["disadvantaged"] == 1]["advanced"]
    or_pooled = _haldane_or(exp, ref)
    rows.append({
        "group": "pooled_disadvantaged",
        "or_total": or_pooled,
        "or_direct": or_pooled,
        "mediated_share_log_odds": 0.0,
        "e_value_direct": evalue_for_rr(or_pooled),
        "term": "disadvantaged",
        "total_fit_method": "fast_smoke_rate_or",
        "direct_fit_method": "fast_smoke_rate_or",
    })
    for group in DISADVANTAGED + ["East-Asian"]:
        tmp = work[work["ethnicity"].isin(["French", group])].copy()
        if tmp["ethnicity"].nunique() < 2:
            continue
        or_g = _haldane_or(tmp[tmp["ethnicity"] == group]["advanced"], tmp[tmp["ethnicity"] == "French"]["advanced"])
        rows.append({
            "group": group,
            "or_total": or_g,
            "or_direct": or_g,
            "mediated_share_log_odds": 0.0,
            "e_value_direct": evalue_for_rr(or_g),
            "term": "exposed",
            "total_fit_method": "fast_smoke_rate_or",
            "direct_fit_method": "fast_smoke_rate_or",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = prepare_common(read_table(args.input))
    validate_columns(df, REQUIRED_SCREEN_COLUMNS)
    work = df.dropna(subset=["advanced", "fit_q", "ethnicity", "gender", "candidate_id"]).copy()

    if os.environ.get("STAGEAUDIT_FAST", "0") == "1":
        out = fast_smoke_table(work)
        write_table(out, args.out)
        print(f"wrote {args.out} (fast synthetic smoke approximation)")
        return

    from stageaudit.stats import fit_logit, odds_ratio_from_two_models

    work["disadvantaged"] = work["ethnicity"].isin(DISADVANTAGED).astype(int)
    total = fit_logit("advanced ~ disadvantaged + C(gender)", work, cluster="candidate_id")
    direct = fit_logit("advanced ~ disadvantaged + fit_q + C(gender)", work, cluster="candidate_id")
    pooled = odds_ratio_from_two_models(total, direct, "disadvantaged")
    pooled["group"] = "pooled_disadvantaged"
    pooled["e_value_direct"] = evalue_for_rr(pooled["or_direct"])

    rows = [pooled]
    for group in DISADVANTAGED + ["East-Asian"]:
        tmp = work[work["ethnicity"].isin(["French", group])].copy()
        if tmp["ethnicity"].nunique() < 2:
            continue
        tmp["exposed"] = (tmp["ethnicity"] == group).astype(int)
        total_g = fit_logit("advanced ~ exposed + C(gender)", tmp, cluster="candidate_id")
        direct_g = fit_logit("advanced ~ exposed + fit_q + C(gender)", tmp, cluster="candidate_id")
        row = odds_ratio_from_two_models(total_g, direct_g, "exposed")
        row["group"] = group
        row["e_value_direct"] = evalue_for_rr(row["or_direct"])
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["group", "or_total", "or_direct", "mediated_share_log_odds", "e_value_direct", "term", "total_fit_method", "direct_fit_method"]
    out = out[[c for c in cols if c in out.columns]]
    write_table(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
