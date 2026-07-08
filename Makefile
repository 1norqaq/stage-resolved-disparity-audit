.EXPORT_ALL_VARIABLES:

DATA ?= data/linked_applications.parquet
PYTHON ?= python
BOOTSTRAP ?= 1000
DEMO_N ?= 5000
DEMO_BOOTSTRAP ?= 10
SMOKE_N ?= 2500
SMOKE_BOOTSTRAP ?= 2

.PHONY: all demo smoke full-synthetic validate clean expected

all:
	$(PYTHON) scripts/run_pipeline.py --input $(DATA) --bootstrap $(BOOTSTRAP) --python $(PYTHON)

validate:
	$(PYTHON) scripts/00_validate_inputs.py --input $(DATA)

# Fast reviewer smoke test. It checks schema/output generation on synthetic data;
# it does not reproduce, validate, or approximate paper estimates.
smoke:
	$(PYTHON) scripts/smoke_test.py --n $(SMOKE_N) --bootstrap $(SMOKE_BOOTSTRAP)

# Default synthetic demo aliases the robust smoke test so it is reliable out of the box.
demo:
	$(PYTHON) scripts/smoke_test.py --n $(DEMO_N) --bootstrap $(DEMO_BOOTSTRAP)

# Optional script-by-script synthetic pipeline for debugging the analysis scripts.
full-synthetic:
	$(PYTHON) scripts/make_synthetic_data.py --out data/synthetic_linked_applications.csv --n $(DEMO_N) --seed 2027
	$(PYTHON) scripts/run_pipeline.py --input data/synthetic_linked_applications.csv --bootstrap $(DEMO_BOOTSTRAP) --fast --python $(PYTHON)

expected: smoke
	$(PYTHON) scripts/10_write_run_manifest.py --input results/tables --out results/run_manifest.json

clean:
	rm -f results/tables/*.csv results/tables/*.json results/figures/*.png results/figures/*.pdf results/figures/*.txt results/run_manifest.json
