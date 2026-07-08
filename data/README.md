# Data Directory

Place governed input files here when running the analysis locally. Do not commit real applicant data.

Expected governed input:

```text
data/linked_applications.parquet
```

For code-path testing, generate synthetic data:

```bash
python scripts/make_synthetic_data.py --out data/synthetic_linked_applications.csv --n 5000 --seed 2027
```
