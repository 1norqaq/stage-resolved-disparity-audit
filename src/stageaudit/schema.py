from __future__ import annotations

import pandas as pd

REQUIRED_SCREEN_COLUMNS = [
    "application_id",
    "candidate_id",
    "job_id",
    "ethnicity",
    "gender",
    "fit_q",
    "advanced",
]

REQUIRED_HIRE_COLUMNS = REQUIRED_SCREEN_COLUMNS + ["hired"]

OPTIONAL_CONTROL_COLUMNS = [
    "ethnicity_posterior",
    "age_band",
    "region",
    "application_region",
    "application_month",
    "job_family",
    "job_category",
    "experience_duration",
    "education_duration",
    "text_length",
    "prequalified",
    "interview_shortlist",
    "offered",
]


def validate_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    return missing


def prepare_common(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["ethnicity", "gender", "age_band", "region", "application_region", "application_month", "job_family", "job_category"]:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("Unknown")
    for col in ["advanced", "hired", "prequalified", "interview_shortlist", "offered"]:
        if col in out.columns:
            out[col] = out[col].astype(int)
    if "fit_q" in out.columns:
        out["fit_q"] = pd.to_numeric(out["fit_q"], errors="coerce")
    return out


def embedding_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("emb_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)
