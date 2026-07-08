# Reviewer quickstart

This archive contains two reproducibility modes.

## 1. Smoke/demo mode: always public, fast, synthetic only

Run:

```bash
make smoke
# or
make demo
```

This creates `data/synthetic_linked_applications.csv`, writes all expected result-table filenames under `results/tables/`, writes `results/run_manifest.json`, and adds a text placeholder in `results/figures/`.

The smoke/demo outputs are **not evidence for the paper**. They are deliberately labelled with `synthetic_smoke_*` methods where the governed-data estimators would be inappropriate or unstable on tiny public synthetic data. Their purpose is to verify that the package layout, schemas, result filenames, and environment are usable during review.

On the provided environment, `make smoke` completes in a few seconds.

## 2. Governed-data mode: reproduces paper analyses when data are available

Place the governed application-level table at:

```text
data/linked_applications.parquet
```

Then run:

```bash
make all DATA=data/linked_applications.parquet BOOTSTRAP=1000
```

This runs the analysis scripts rather than the smoke approximations. The governed table is not redistributed because it contains sensitive recruitment records and inferred protected attributes.

## Optional script-by-script synthetic debugging

```bash
make full-synthetic
```

This runs the analysis scripts on synthetic data with fast/safe fallbacks. It is useful for debugging but should not be cited as paper reproduction. Sparse synthetic data can still produce placeholder or regularized estimates, which is why `make smoke`/`make demo` is the default reviewer path.
