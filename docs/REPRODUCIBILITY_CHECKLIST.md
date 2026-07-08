# Reproducibility Checklist

## Materials

- [x] README describes the governed-data analysis workflow and the public smoke/demo workflow.
- [x] Codebook lists all required columns.
- [x] Environment file and requirements file are included.
- [x] License is included.
- [x] Data availability statement explains why raw data are not public.
- [x] Synthetic data generator is included for script-level debugging.
- [x] `make smoke` / `make demo` generate a public synthetic schema check and all expected result-table filenames.
- [x] Scripts write governed-data tables and figures to `results/`.

## Data handling

- [x] No real applicant rows are committed.
- [x] No names, CV text, geolocation coordinates, linkage keys, or employer identifiers are committed.
- [x] Small-cell disclosure review is documented as required before releasing aggregate results.
- [x] Governed data are stored outside the public repository.

## Statistical workflow for governed-data reproduction

- [x] Candidate-level clustering is used for repeated applications.
- [x] Screen-gate models condition on `fit_q` and gender.
- [x] Multiple origin contrasts are adjusted with Benjamini-Hochberg correction.
- [x] Selection-rate ratios are reported on the rate scale by g-computation.
- [x] Hire-gate uncertainty is estimated by whole-candidate-cluster bootstrap.
- [x] Sparse intermediate gates are estimated with bias-reduced or penalized logistic models, with finite fallbacks for separated data.
- [x] Mediation reports total effect, direct effect, mediated share through `fit_q`, and E-values.
- [x] Representation probes use out-of-fold evaluation.
- [x] Content-leakage models report AUC and permutation importance.
- [x] Robustness checks include confidence thresholds and sorting controls.

## Public synthetic smoke/demo limits

- [x] Smoke/demo outputs are explicitly labelled as synthetic or placeholder methods.
- [x] Smoke/demo outputs are not represented as reproducing the paper's empirical estimates.
- [x] A `results/run_manifest.json` file records the smoke/demo run.
