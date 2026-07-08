from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from _bootstrap_path import ROOT  # noqa: F401
from stageaudit.io import write_table

GROUPS = [
    "French",
    "Muslim",
    "African",
    "Indian-subcontinent",
    "East-Asian",
    "British",
    "Italian",
    "Hispanic",
    "East-European",
    "Jewish",
    "Nordic",
    "Germanic",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--emb-dim", type=int, default=16, help="Number of synthetic embedding columns to generate; real data may have 1024.")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_candidates = max(500, args.n // 5)
    candidate_ids = [f"cand_{i:05d}" for i in range(n_candidates)]
    job_ids = [f"job_{i:05d}" for i in range(max(200, args.n // 10))]

    group_probs = np.array([0.22, 0.21, 0.17, 0.06, 0.04, 0.14, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02])
    group_probs = group_probs / group_probs.sum()
    candidate_group = dict(zip(candidate_ids, rng.choice(GROUPS, size=n_candidates, p=group_probs)))

    rows = []
    group_screen_effect = {
        "French": 0.0,
        "Muslim": -0.42,
        "African": -0.35,
        "Indian-subcontinent": -0.45,
        "East-Asian": -0.25,
        "British": -0.06,
        "Italian": -0.03,
        "Hispanic": -0.12,
        "East-European": -0.40,
        "Jewish": -0.40,
        "Nordic": -0.34,
        "Germanic": 0.10,
    }
    group_hire_effect = {g: 0.0 for g in GROUPS}
    group_hire_effect.update({"Muslim": -0.60, "African": -0.45, "Indian-subcontinent": -0.30})

    for i in range(args.n):
        cid = rng.choice(candidate_ids)
        gid = candidate_group[cid]
        fit = np.clip(rng.normal(0.0, 1.0), -3, 3)
        gender = rng.choice(["Female", "Male", "Unknown"], p=[0.43, 0.45, 0.12])
        job = rng.choice(job_ids)
        job_family = rng.choice(["Sales", "Operations", "Engineering", "Admin", "Service"])
        job_category = f"cat_{rng.integers(1, 31):02d}"
        region = rng.choice(["North", "South", "East", "West", "Central", "Unknown"], p=[.18, .19, .18, .17, .20, .08])
        month = f"2026-{rng.integers(1, 13):02d}"
        # Keep the synthetic outcome rates high enough for reviewers to run
        # smoke tests without separation, while preserving the qualitative
        # screen-stage and hire-stage disparities used to exercise code paths.
        lp_adv = -2.05 + 0.35 * fit + group_screen_effect[gid] + (0.04 if gender == "Female" else 0.0)
        p_adv = 1 / (1 + np.exp(-lp_adv))
        advanced = int(rng.uniform() < p_adv)
        lp_hire = -1.35 + 0.20 * fit + group_hire_effect[gid]
        p_hire = 1 / (1 + np.exp(-lp_hire)) if advanced else 0.0
        hired = int(advanced and rng.uniform() < p_hire)
        preq = int(advanced and rng.uniform() < (0.42 + 0.20 * p_hire))
        interview = int(preq and rng.uniform() < 0.55)
        offered = int(interview and rng.uniform() < 0.38)
        row = {
            "application_id": f"app_{i:07d}",
            "candidate_id": cid,
            "job_id": job,
            "ethnicity": gid,
            "ethnicity_posterior": float(np.clip(rng.beta(7, 3), 0.05, 0.99)),
            "gender": gender,
            "age_band": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+", "Unknown"]),
            "region": region,
            "application_region": region,
            "application_month": month,
            "job_family": job_family,
            "job_category": job_category,
            "fit_q": fit,
            "experience_duration": float(np.clip(rng.normal(6, 4), 0, 30)),
            "education_duration": float(np.clip(rng.normal(4, 2), 0, 12)),
            "text_length": int(np.clip(rng.normal(2500, 900), 200, 8000)),
            "advanced": advanced,
            "hired": hired,
            "prequalified": preq,
            "interview_shortlist": interview,
            "offered": offered,
        }
        for j in range(args.emb_dim):
            # Synthetic embeddings contain only a weak group cue; they are for
            # code-path tests and are not used to substantiate paper claims.
            row[f"emb_{j}"] = float(rng.normal(0, 1) + (0.1 if gid == "French" and j == 0 else 0))
        rows.append(row)

    out = pd.DataFrame(rows)
    write_table(out, args.out)
    print(f"wrote {len(out):,} rows to {args.out}")


if __name__ == "__main__":
    main()
