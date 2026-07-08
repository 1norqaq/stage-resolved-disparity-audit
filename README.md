# Where Disparity Enters: Reproducibility Materials

This repository contains the analysis code and documentation needed to reproduce the quantitative results reported in *Where Disparity Enters: Stage-Resolved Attribution of Ethnic Disparate Impact in AI-Assisted Hiring*.

The repository is designed for double-anonymous review. It contains no raw applicant records, names, CV text, employer identifiers, credentials, private paths, or deployment-specific secrets. The private linked application-level table is governed by a data-use agreement and is not redistributed here.

## What is included

- `scripts/`: executable analysis scripts for validation, screen-gate disparate-impact models, standardized selection-rate ratios, hire-gate bootstrap intervals, sparse intermediate-stage models, mediation/E-value analysis, representation probes, content-leakage analysis, and robustness checks.
- `src/stageaudit/`: reusable helper functions for schema validation, logistic modeling, bootstrap resampling, standardized rate ratios, E-values, and plotting.
- `docs/`: data dictionary, data availability statement, reviewer quickstart, ethics and privacy statement, anonymization notes, and reproducibility checklist.
- `data/`: expected location for governed input files. This directory contains only documentation and optional synthetic data generated locally.
- `results/`: output directory for tables and figures.

## Data availability

The empirical analysis uses a linked application-level table derived from a production applicant-tracking-system export. The table contains sensitive personal data and inferred protected attributes. It cannot be placed in a public repository. See `docs/DATA_AVAILABILITY.md` for the access policy and a description of the released schema.

A public synthetic dataset can be generated for review-time smoke testing:

```bash
make smoke
# or, with more rows
make demo
```

The synthetic outputs are not used to support any claim in the paper and do not reproduce the paper estimates. They check that the repository layout, input schema, output filenames, and result-table schemas are usable without access to the governed data. See `docs/REVIEWER_QUICKSTART.md` for the distinction between smoke/demo mode and governed-data reproduction.

## Expected input

Place the governed analysis table at:

```text
data/linked_applications.parquet
```

Required columns are listed in `docs/CODEBOOK.md`. The scripts also accept `.csv` input via `--input path/to/file.csv`.

The minimal table contains one row per application and includes:

- stable application, candidate, and job identifiers;
- inferred ethnicity label and posterior confidence;
- declared or inferred gender, age band, and region where available;
- candidate-job fit score `fit_q`;
- screen, hire, and intermediate-stage outcomes;
- job family/category, application region, and month controls;
- optional 1024-dimensional profile embedding columns named `emb_0` through `emb_1023`.

## Environment

Create the Python environment with either Conda or pip.

```bash
conda env create -f environment.yml
conda activate wde-stage-audit
```

or

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts were written for Python 3.11. The Firth intermediate-stage analysis can be run with the included Python fallback or with the optional R script if `logistf` is available.

## Reproducing the paper analyses

Run the full pipeline on the governed table:

```bash
make all DATA=data/linked_applications.parquet BOOTSTRAP=1000
```

Run a fast public smoke/demo check on synthetic data:

```bash
make smoke
make demo
```

For script-by-script debugging on synthetic data, use:

```bash
make full-synthetic
```

`make smoke` and `make demo` intentionally use lightweight synthetic placeholders for the sparsest downstream pieces; `make all` is the governed-data reproduction path.

Individual steps:

```bash
python scripts/00_validate_inputs.py --input data/linked_applications.parquet
python scripts/01_screen_gate_disparate_impact.py --input data/linked_applications.parquet --out results/tables/screen_gate.csv
python scripts/02_selection_rate_ratios.py --input data/linked_applications.parquet --out results/tables/selection_rate_ratios.csv
python scripts/03_hire_gate_bootstrap.py --input data/linked_applications.parquet --out results/tables/hire_gate_bootstrap.csv --bootstrap 1000
python scripts/04_intermediate_stages.py --input data/linked_applications.parquet --out results/tables/intermediate_stages.csv
python scripts/05_mediation_evalues.py --input data/linked_applications.parquet --out results/tables/mediation_evalues.csv
python scripts/06_representation_probe.py --input data/linked_applications.parquet --out results/tables/representation_probe.csv
python scripts/07_content_leakage.py --input data/linked_applications.parquet --out results/tables/content_leakage.csv
python scripts/08_robustness_checks.py --input data/linked_applications.parquet --outdir results/tables
python scripts/09_make_figures.py --input results/tables --outdir results/figures
```

## Main outputs

The expected governed-data outputs are:

- screen-gate adjusted odds ratios by origin group;
- standardized selection-rate ratios from g-computation;
- candidate-clustered bootstrap intervals for the hire gate among advanced applications;
- Firth or penalized sparse-stage estimates for pre-qualification, interview/shortlist, and offer milestones;
- total/direct effect odds ratios, mediated share through `fit_q`, and E-values;
- cross-validated macro-AUC for protected-attribute recovery from embeddings;
- gradient-boosted content-leakage AUC and permutation importance;
- robustness tables for stricter name-inference confidence, job/location/month controls, and inverse-propensity reweighting.

In smoke/demo mode, the same output filenames are created, but rows are marked with `synthetic_smoke_*` or placeholder methods where appropriate. These files are only a schema and environment check.

## Reproducibility notes

- Candidate-level clustering is used whenever an applicant contributes multiple applications.
- Bootstrap resampling is performed at the candidate-cluster level, not row level.
- The French group is the reference category for origin comparisons.
- The headline screen model conditions on `fit_q` and gender, with false-discovery-rate correction across origin contrasts.
- Selection-rate ratios are computed on the rate scale by standardizing over the empirical distribution of non-group covariates.
- Downstream hire models are estimated among applications that advanced past the CV screen.
- Intermediate milestones are sparse and should be interpreted with the uncertainty intervals reported by the script.

## License

Code is released under the MIT License. See `LICENSE`.

The governed empirical dataset is not covered by the MIT License and is not redistributed. Synthetic data generated by `scripts/make_synthetic_data.py` is released under CC0; see `docs/DATA_USE_TERMS.md`.
