from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from _bootstrap_path import ROOT  # noqa: F401
from stageaudit.io import read_table, write_table
from stageaudit.schema import prepare_common

NUMERIC = ["fit_q", "experience_duration", "education_duration", "text_length"]


def rank_auc(y: pd.Series, score: pd.Series) -> float:
    y = y.astype(int).to_numpy()
    score = pd.Series(score).rank(method="average").to_numpy()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = score[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fast_auc(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["advanced"]).copy()
    nums = [c for c in NUMERIC if c in work.columns]
    if nums:
        z = []
        for c in nums:
            x = pd.to_numeric(work[c], errors="coerce").fillna(work[c].median())
            sd = x.std() or 1.0
            z.append((x - x.mean()) / sd)
        content_score = sum(z)
    else:
        content_score = pd.Series(np.zeros(len(work)), index=work.index)
    auc_content = rank_auc(work["advanced"], content_score)
    eth_score = content_score.copy()
    if "ethnicity" in work.columns:
        rates = work.groupby("ethnicity")["advanced"].mean()
        eth_score = eth_score + work["ethnicity"].map(rates).fillna(work["advanced"].mean()).to_numpy()
    auc_eth = rank_auc(work["advanced"], eth_score)
    rows = [
        {"metric": "auc_content_only", "value": auc_content, "model": "fast_smoke_rank_auc"},
        {"metric": "auc_content_plus_ethnicity", "value": auc_eth, "model": "fast_smoke_rank_auc"},
        {"metric": "auc_gain", "value": auc_eth - auc_content, "model": "fast_smoke_rank_auc"},
    ]
    for c in nums:
        rows.append({"metric": c, "value": abs(rank_auc(work["advanced"], work[c]) - 0.5), "model": "fast_smoke_rank_importance"})
    if "ethnicity" in work.columns:
        rows.append({"metric": "ethnicity", "value": abs(auc_eth - auc_content), "model": "fast_smoke_rank_importance"})
    return pd.DataFrame(rows)


def fit_auc(df: pd.DataFrame, features: list[str], seed: int):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    work = df.dropna(subset=["advanced"] + [f for f in features if f in df.columns]).copy()
    y = work["advanced"].astype(int)
    X = work[features]
    cat = [c for c in features if c not in NUMERIC]
    num = [c for c in features if c in NUMERIC]
    prep = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    clf = Pipeline([
        ("prep", prep),
        ("gb", GradientBoostingClassifier(random_state=seed)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    perm = permutation_importance(clf, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=seed)
    imp = pd.DataFrame({"feature": features, "permutation_importance": perm.importances_mean})
    return float(auc), imp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    df = prepare_common(read_table(args.input))
    if os.environ.get("STAGEAUDIT_FAST", "0") == "1":
        out = fast_auc(df)
        write_table(out, args.out)
        print(f"wrote {args.out} (fast synthetic smoke approximation)")
        return

    content_features = [c for c in NUMERIC if c in df.columns]
    with_ethnicity = content_features + (["ethnicity"] if "ethnicity" in df.columns else [])
    auc_content, imp_content = fit_auc(df, content_features, args.seed)
    auc_eth, imp_eth = fit_auc(df, with_ethnicity, args.seed)
    imp_eth["model"] = "content_plus_ethnicity"
    imp_content["model"] = "content_only"
    summary = pd.DataFrame([
        {"metric": "auc_content_only", "value": auc_content, "model": "content_only"},
        {"metric": "auc_content_plus_ethnicity", "value": auc_eth, "model": "content_plus_ethnicity"},
        {"metric": "auc_gain", "value": auc_eth - auc_content, "model": "summary"},
    ])
    out = pd.concat([
        summary,
        imp_content.rename(columns={"feature": "metric", "permutation_importance": "value"})[["metric", "value", "model"]],
        imp_eth.rename(columns={"feature": "metric", "permutation_importance": "value"})[["metric", "value", "model"]],
    ], ignore_index=True)
    write_table(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
