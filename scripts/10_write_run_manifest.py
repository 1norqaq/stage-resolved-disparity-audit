from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import pandas as pd


def summarize_table(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    df = pd.read_csv(path)
    summary = {"exists": True, "rows": int(len(df)), "columns": list(df.columns)}
    numeric = df.select_dtypes(include="number")
    if len(numeric.columns):
        summary["numeric_means"] = {c: float(numeric[c].mean()) for c in numeric.columns[:12]}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing result tables")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    indir = Path(args.input)
    files = sorted(indir.glob("*.csv"))
    manifest = {
        "purpose": "Smoke-test manifest for synthetic data only; not paper evidence.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tables": {p.name: summarize_table(p) for p in files},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
