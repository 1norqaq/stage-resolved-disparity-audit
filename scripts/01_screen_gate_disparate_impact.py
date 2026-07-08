from __future__ import annotations

import argparse
import re
from _bootstrap_path import ROOT  # noqa: F401
from stageaudit.io import read_table, write_table
from stageaudit.schema import REQUIRED_SCREEN_COLUMNS, prepare_common, validate_columns
from stageaudit.stats import add_fdr, coefficient_table, fit_logit


def parse_group(term: str) -> str:
    m = re.search(r"\[T\.(.*)\]", term)
    return m.group(1) if m else term


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
    tab = coefficient_table(result, "C(ethnicity")
    tab["group"] = tab["term"].map(parse_group)
    tab = add_fdr(tab)
    tab["fails_four_fifths_or"] = (tab["or"] < 0.8) & (tab["p_fdr"] < 0.05)
    tab = tab[["group", "or", "ci_low", "ci_high", "p_value", "p_fdr", "reject_fdr_0_05", "fails_four_fifths_or", "term"]]
    write_table(tab, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
