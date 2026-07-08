from __future__ import annotations

import argparse
from _bootstrap_path import ROOT  # noqa: F401
from stageaudit.io import read_table, write_table
from stageaudit.schema import REQUIRED_SCREEN_COLUMNS, prepare_common, validate_columns
from stageaudit.stats import fit_logit, standardized_selection_rate_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", default="French")
    args = parser.parse_args()

    df = prepare_common(read_table(args.input))
    validate_columns(df, REQUIRED_SCREEN_COLUMNS)
    keep = ["advanced", "fit_q", "ethnicity", "gender", "candidate_id"]
    df = df.dropna(subset=keep).copy()
    formula = f"advanced ~ fit_q + C(ethnicity, Treatment(reference='{args.reference}')) + C(gender)"
    result = fit_logit(formula, df, cluster="candidate_id")
    tab = standardized_selection_rate_ratio(df, result, group_col="ethnicity", ref=args.reference, outcome_col="advanced")
    tab["fails_four_fifths_srr"] = tab["srr"] < 0.8
    write_table(tab, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
