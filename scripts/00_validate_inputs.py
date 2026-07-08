from __future__ import annotations

import argparse
import json
from _bootstrap_path import ROOT  # noqa: F401
from stageaudit.io import read_table
from stageaudit.schema import REQUIRED_SCREEN_COLUMNS, REQUIRED_HIRE_COLUMNS, OPTIONAL_CONTROL_COLUMNS, embedding_columns, validate_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = read_table(args.input)
    validate_columns(df, REQUIRED_SCREEN_COLUMNS)
    missing_hire = [c for c in REQUIRED_HIRE_COLUMNS if c not in df.columns]
    optional_present = [c for c in OPTIONAL_CONTROL_COLUMNS if c in df.columns]
    emb_cols = embedding_columns(df)
    report = {
        "rows": int(len(df)),
        "candidates": int(df["candidate_id"].nunique()) if "candidate_id" in df else None,
        "jobs": int(df["job_id"].nunique()) if "job_id" in df else None,
        "missing_hire_columns": missing_hire,
        "optional_columns_present": optional_present,
        "embedding_columns": len(emb_cols),
        "advanced_rate": float(df["advanced"].mean()) if "advanced" in df else None,
        "hire_rate": float(df["hired"].mean()) if "hired" in df else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
