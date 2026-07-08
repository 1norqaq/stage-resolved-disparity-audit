from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from _bootstrap_path import ROOT  # noqa: F401
from stageaudit.io import read_table, write_table
from stageaudit.schema import embedding_columns, prepare_common

ATTRIBUTES = ["ethnicity", "gender", "region", "age_band"]


def macro_auc(X: np.ndarray, y: pd.Series, seed: int) -> float:
    y = y.astype("string").fillna("Unknown")
    classes = sorted(y.unique())
    if len(classes) < 2:
        return float("nan")
    counts = y.value_counts()
    min_count = int(counts.min())
    if min_count < 3:
        return float("nan")
    n_splits = min(3, min_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train, test in cv.split(X, y):
        clf = LogisticRegression(max_iter=1000, n_jobs=None)
        clf.fit(X[train], y.iloc[train])
        proba = clf.predict_proba(X[test])
        class_order = list(clf.classes_)
        y_bin = label_binarize(y.iloc[test], classes=class_order)
        try:
            if len(class_order) == 2:
                auc = roc_auc_score(y_bin, proba[:, 1])
            else:
                auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
            aucs.append(float(auc))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    df = prepare_common(read_table(args.input))
    emb_cols = embedding_columns(df)
    rows = []
    if not emb_cols:
        out = pd.DataFrame({"attribute": ATTRIBUTES, "embedding_macro_auc": [float("nan")] * len(ATTRIBUTES), "note": ["embedding columns absent"] * len(ATTRIBUTES)})
        write_table(out, args.out)
        print(f"wrote {args.out}")
        return
    X = df[emb_cols].fillna(0.0).to_numpy(dtype=float)
    for attr in ATTRIBUTES:
        if attr in df.columns:
            rows.append({"attribute": attr, "embedding_macro_auc": macro_auc(X, df[attr], args.seed), "n": len(df)})
    write_table(pd.DataFrame(rows), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
