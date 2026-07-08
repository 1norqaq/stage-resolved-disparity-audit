from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = [
    "French", "Muslim", "African", "Indian-subcontinent", "East-Asian", "British",
    "Italian", "Hispanic", "East-European", "Jewish", "Nordic", "Germanic",
]
DISADVANTAGED = ["Muslim", "African", "Indian-subcontinent"]


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def synth(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_candidates = max(300, n // 5)
    candidate_ids = np.array([f"cand_{i:05d}" for i in range(n_candidates)])
    job_ids = np.array([f"job_{i:05d}" for i in range(max(100, n // 10))])
    group_probs = np.array([0.22, 0.21, 0.17, 0.06, 0.04, 0.14, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02])
    group_probs /= group_probs.sum()
    cand_group = dict(zip(candidate_ids, rng.choice(GROUPS, size=n_candidates, p=group_probs)))
    screen_eff = {g: 0.0 for g in GROUPS}
    screen_eff.update({"Muslim": -0.42, "African": -0.35, "Indian-subcontinent": -0.45, "East-Asian": -0.25, "East-European": -0.40, "Jewish": -0.40, "Nordic": -0.34})
    hire_eff = {g: 0.0 for g in GROUPS}
    hire_eff.update({"Muslim": -0.60, "African": -0.45, "Indian-subcontinent": -0.30})
    rows = []
    for i in range(n):
        cid = rng.choice(candidate_ids)
        g = cand_group[cid]
        fit = np.clip(rng.normal(), -3, 3)
        gender = rng.choice(["Female", "Male", "Unknown"], p=[0.43, 0.45, 0.12])
        lp_adv = -2.05 + 0.35 * fit + screen_eff[g] + (0.04 if gender == "Female" else 0.0)
        p_adv = 1 / (1 + np.exp(-lp_adv))
        advanced = int(rng.uniform() < p_adv)
        lp_hire = -1.35 + 0.20 * fit + hire_eff[g]
        p_hire = 1 / (1 + np.exp(-lp_hire)) if advanced else 0.0
        hired = int(advanced and rng.uniform() < p_hire)
        preq = int(advanced and rng.uniform() < (0.42 + 0.20 * p_hire))
        interview = int(preq and rng.uniform() < 0.55)
        offered = int(interview and rng.uniform() < 0.38)
        rows.append({
            "application_id": f"app_{i:07d}", "candidate_id": cid, "job_id": rng.choice(job_ids),
            "ethnicity": g, "ethnicity_posterior": float(np.clip(rng.beta(7, 3), .05, .99)),
            "gender": gender, "age_band": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+", "Unknown"]),
            "region": rng.choice(["North", "South", "East", "West", "Central", "Unknown"]),
            "application_region": rng.choice(["North", "South", "East", "West", "Central", "Unknown"]),
            "application_month": f"2026-{rng.integers(1, 13):02d}",
            "job_family": rng.choice(["Sales", "Operations", "Engineering", "Admin", "Service"]),
            "job_category": f"cat_{rng.integers(1, 31):02d}",
            "fit_q": fit,
            "experience_duration": float(np.clip(rng.normal(6, 4), 0, 30)),
            "education_duration": float(np.clip(rng.normal(4, 2), 0, 12)),
            "text_length": int(np.clip(rng.normal(2500, 900), 200, 8000)),
            "advanced": advanced, "hired": hired, "prequalified": preq,
            "interview_shortlist": interview, "offered": offered,
            **{f"emb_{j}": float(rng.normal() + (0.1 if g == "French" and j == 0 else 0.0)) for j in range(16)},
        })
    return pd.DataFrame(rows)


def log_or(exp_y: pd.Series, ref_y: pd.Series) -> tuple[float, float]:
    a = float(exp_y.sum()) + 0.5; b = float(len(exp_y) - exp_y.sum()) + 0.5
    c = float(ref_y.sum()) + 0.5; d = float(len(ref_y) - ref_y.sum()) + 0.5
    return float(np.log((a / b) / (c / d))), float(np.sqrt(1/a + 1/b + 1/c + 1/d))


def or_rows(df: pd.DataFrame, outcome: str, reference: str = "French") -> pd.DataFrame:
    ref = df.loc[df.ethnicity == reference, outcome]
    rows = []
    for g in sorted([x for x in df.ethnicity.unique() if x != reference]):
        beta, se = log_or(df.loc[df.ethnicity == g, outcome], ref)
        rows.append({"group": g, "or": np.exp(beta), "ci_low": np.exp(beta - 1.96*se), "ci_high": np.exp(beta + 1.96*se),
                     "p_value": np.nan, "p_fdr": np.nan, "reject_fdr_0_05": False, "term": f"ethnicity={g}", "fit_method": "synthetic_smoke_rate_or"})
    return pd.DataFrame(rows)


def evalue(rr: float) -> float:
    if rr < 1: rr = 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--bootstrap", type=int, default=2)
    ap.add_argument("--data", default="data/synthetic_linked_applications.csv")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = Path.cwd()
    results = root / args.results
    tables = results / "tables"
    figures = results / "figures"
    df = synth(args.n, args.seed)
    write(df, root / args.data)

    screen = or_rows(df, "advanced")
    screen["fails_four_fifths_or"] = screen["or"] < 0.8
    write(screen, tables / "screen_gate.csv")

    ref_rate = df.loc[df.ethnicity == "French", "advanced"].mean()
    srr = df.groupby("ethnicity")["advanced"].mean().reset_index(name="standardized_rate")
    srr = srr[srr.ethnicity != "French"].rename(columns={"ethnicity": "group"})
    srr["reference_rate"] = ref_rate; srr["srr"] = srr["standardized_rate"] / ref_rate; srr["fails_four_fifths_srr"] = srr.srr < 0.8
    srr["fit_method"] = "synthetic_smoke_rate_ratio"
    write(srr, tables / "selection_rate_ratios.csv")

    adv = df[df.advanced == 1]
    hire = or_rows(adv, "hired") if len(adv) else pd.DataFrame()
    hire["successful_bootstraps"] = args.bootstrap; hire["requested_bootstraps"] = args.bootstrap; hire["failed_bootstraps_total"] = 0
    write(hire, tables / "hire_gate_bootstrap.csv")

    interm = []
    for m in ["prequalified", "interview_shortlist", "offered"]:
        work = df[df.ethnicity.isin(["French"] + DISADVANTAGED)]
        beta, se = log_or(work.loc[work.ethnicity.isin(DISADVANTAGED), m], work.loc[work.ethnicity == "French", m])
        interm.append({"milestone": m, "estimate": beta, "std_error": se, "or": np.exp(beta), "ci_low": np.exp(beta-1.96*se), "ci_high": np.exp(beta+1.96*se), "n": len(work), "events": int(work[m].sum()), "fit_method": "synthetic_smoke_rate_or"})
    interm = pd.DataFrame(interm)
    interm["eb_estimate"] = interm.estimate.mean(); interm["eb_or"] = np.exp(interm.eb_estimate); interm["tau2"] = 0.0
    interm["common_estimate"] = interm.estimate.mean(); interm["common_or"] = np.exp(interm.common_estimate)
    interm["common_ci_low"] = np.exp(interm.common_estimate - 1.96 * interm.std_error.mean()); interm["common_ci_high"] = np.exp(interm.common_estimate + 1.96 * interm.std_error.mean())
    write(interm, tables / "intermediate_stages.csv")

    pooled = df.copy(); pooled["disadvantaged"] = pooled.ethnicity.isin(DISADVANTAGED)
    beta, _ = log_or(pooled.loc[pooled.disadvantaged, "advanced"], pooled.loc[~pooled.disadvantaged, "advanced"])
    med = pd.DataFrame([{"group": "pooled_disadvantaged", "or_total": np.exp(beta), "or_direct": np.exp(beta), "mediated_share_log_odds": 0.0, "e_value_direct": evalue(np.exp(beta)), "term": "disadvantaged", "total_fit_method": "synthetic_smoke_rate_or", "direct_fit_method": "synthetic_smoke_rate_or"}])
    write(med, tables / "mediation_evalues.csv")

    probe = pd.DataFrame({"attribute": ["ethnicity", "gender", "region", "age_band"], "embedding_macro_auc": [0.52, 0.50, 0.53, 0.50], "n": len(df), "fit_method": "synthetic_smoke_placeholder"})
    write(probe, tables / "representation_probe.csv")

    leak = pd.DataFrame([{"metric": "auc_content_only", "value": 0.52, "model": "synthetic_smoke_placeholder"}, {"metric": "auc_content_plus_ethnicity", "value": 0.54, "model": "synthetic_smoke_placeholder"}, {"metric": "auc_gain", "value": 0.02, "model": "synthetic_smoke_placeholder"}])
    write(leak, tables / "content_leakage.csv")

    conf = pd.concat([screen.assign(specification="posterior_ge_0.0", n=len(df)), screen.assign(specification="posterior_ge_0.5", n=int((df.ethnicity_posterior>=0.5).sum()))], ignore_index=True)
    write(conf, tables / "robustness_confidence_thresholds.csv")
    write(screen.assign(specification="sorting_controls_synthetic_smoke_placeholder", n=len(df)), tables / "robustness_sorting_controls.csv")

    figures.mkdir(parents=True, exist_ok=True)
    (figures / "SMOKE_FIGURES_SKIPPED.txt").write_text("Synthetic smoke mode generated tables only. Run make all on governed data for figures.\n")
    manifest = {"purpose": "Synthetic smoke test only; not paper evidence.", "python": platform.python_version(), "rows": len(df), "tables": sorted(p.name for p in tables.glob('*.csv'))}
    (results / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"smoke test completed: wrote {len(df):,} synthetic rows and {len(manifest['tables'])} tables")


if __name__ == "__main__":
    main()
