from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import expit
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

try:
    import patsy
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - imports are checked at runtime when fallback is used
    patsy = None
    LogisticRegression = None


@dataclass
class RidgeLogitResult:
    """Small statsmodels-like wrapper for robust ridge-logit fallback.

    The wrapper exposes the attributes used by the analysis scripts
    (`params`, `bse`, `pvalues`, and `predict`).  It is intentionally
    conservative: standard errors are approximate Hessian/Wald intervals and
    are used only when maximum-likelihood logit is unstable on sparse/synthetic
    data.  The governed-data paper results should still be produced with the
    default statsmodels fit whenever it converges.
    """

    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    design_info: object
    formula: str
    method: str = "ridge-logit-fallback"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if patsy is None:
            raise RuntimeError("patsy is required for RidgeLogitResult.predict")
        X = patsy.build_design_matrices([self.design_info], df, return_type="dataframe")[0]
        X = X.reindex(columns=self.params.index, fill_value=0.0)
        return expit(np.asarray(X, dtype=float) @ self.params.to_numpy())


def _ridge_logit_fit(formula: str, df: pd.DataFrame, *, c: float = 20.0, max_iter: int = 2000) -> RidgeLogitResult:
    """Fit finite, weakly regularized logistic regression using a formula.

    This is a fallback for separation/singular-Hessian cases common in small
    synthetic data and cluster-bootstrap resamples.  It does not provide
    cluster-robust SEs.  It keeps the code path runnable while marking the
    method in `result.method`.
    """
    if patsy is None or LogisticRegression is None:
        raise RuntimeError("patsy and scikit-learn are required for ridge-logit fallback")
    y_df, X_df = patsy.dmatrices(formula, df, return_type="dataframe")
    y = np.asarray(y_df).ravel().astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Outcome has a single class; logit is not identifiable")
    X = np.asarray(X_df, dtype=float)
    clf = LogisticRegression(
        C=c,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=max_iter,
        n_jobs=None,
    )
    clf.fit(X, y)
    beta = clf.coef_.ravel()
    p = clf.predict_proba(X)[:, 1]
    w = np.clip(p * (1.0 - p), 1e-8, None)
    # Approximate penalized Hessian inverse.  Do not penalize the intercept.
    ridge = np.eye(X.shape[1]) / c
    if "Intercept" in X_df.columns:
        ridge[list(X_df.columns).index("Intercept"), list(X_df.columns).index("Intercept")] = 0.0
    hess = (X.T * w) @ X + ridge
    cov = np.linalg.pinv(hess)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan, dtype=float), where=se > 0)
    pvals = 2 * (1 - norm.cdf(np.abs(z)))
    idx = pd.Index(X_df.columns)
    return RidgeLogitResult(
        params=pd.Series(beta, index=idx),
        bse=pd.Series(se, index=idx),
        pvalues=pd.Series(pvals, index=idx),
        design_info=X_df.design_info,
        formula=formula,
    )


def fit_logit(formula: str, df: pd.DataFrame, cluster: str | None = None, *, fallback: bool = True):
    """Fit a logistic model with a ridge fallback for sparse/separated data.

    The first attempt is the intended statsmodels MLE fit.  If it fails because
    of separation, singular matrices, or non-convergence, the function falls
    back to a finite ridge-logit estimate.  This makes the review package's
    synthetic demo robust without changing the primary governed-data code path
    when MLE is well behaved.
    """
    if os.environ.get("STAGEAUDIT_FAST", "0") == "1":
        return _ridge_logit_fit(formula, df)

    model = smf.logit(formula=formula, data=df)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=ConvergenceWarning)
            warnings.simplefilter("always", category=PerfectSeparationWarning)
            if cluster is None:
                result = model.fit(disp=False, maxiter=300)
            else:
                result = model.fit(
                    disp=False,
                    maxiter=300,
                    cov_type="cluster",
                    cov_kwds={"groups": df[cluster].values},
                )
        problem = any(issubclass(w.category, (ConvergenceWarning, PerfectSeparationWarning)) for w in caught)
        converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
        if fallback and (problem or not converged):
            return _ridge_logit_fit(formula, df)
        return result
    except Exception:
        if not fallback:
            raise
        return _ridge_logit_fit(formula, df)


def fit_logit_ridge(formula: str, df: pd.DataFrame):
    """Explicit ridge-logit fit for bootstrap resamples and smoke tests."""
    return _ridge_logit_fit(formula, df)


def coefficient_table(result, term_prefix: str | None = None) -> pd.DataFrame:
    params = pd.Series(result.params)
    bse = pd.Series(result.bse, index=params.index)
    pvals = pd.Series(result.pvalues, index=params.index)
    rows = []
    for term, beta in params.items():
        if term == "Intercept":
            continue
        if term_prefix and term_prefix not in term:
            continue
        lo = beta - 1.96 * bse[term]
        hi = beta + 1.96 * bse[term]
        rows.append({
            "term": term,
            "estimate": beta,
            "std_error": bse[term],
            "or": float(np.exp(beta)),
            "ci_low": float(np.exp(lo)),
            "ci_high": float(np.exp(hi)),
            "p_value": float(pvals[term]),
            "fit_method": getattr(result, "method", "statsmodels-logit"),
        })
    return pd.DataFrame(rows)


def add_fdr(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    out = df.copy()
    if len(out) == 0:
        out["p_fdr"] = []
        out["reject_fdr_0_05"] = []
        return out
    reject, p_adj, _, _ = multipletests(out[p_col].fillna(1.0).values, alpha=0.05, method="fdr_bh")
    out["p_fdr"] = p_adj
    out["reject_fdr_0_05"] = reject
    return out


def odds_ratio_from_two_models(total_result, direct_result, exposure_term: str) -> dict:
    total_beta = float(total_result.params[exposure_term])
    direct_beta = float(direct_result.params[exposure_term])
    if total_beta == 0:
        mediated_share = np.nan
    else:
        mediated_share = 1.0 - direct_beta / total_beta
    return {
        "term": exposure_term,
        "or_total": float(np.exp(total_beta)),
        "or_direct": float(np.exp(direct_beta)),
        "mediated_share_log_odds": float(mediated_share),
        "total_fit_method": getattr(total_result, "method", "statsmodels-logit"),
        "direct_fit_method": getattr(direct_result, "method", "statsmodels-logit"),
    }


def evalue_for_rr(rr: float) -> float:
    rr = float(rr)
    if rr <= 0:
        return np.nan
    if rr < 1:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def standardized_selection_rate_ratio(df: pd.DataFrame, result, group_col: str, ref: str, outcome_col: str = "advanced") -> pd.DataFrame:
    groups = sorted([g for g in df[group_col].dropna().unique() if g != ref])
    base = df.copy()
    ref_df = base.copy()
    ref_df[group_col] = ref
    ref_rate = float(np.asarray(result.predict(ref_df)).mean())
    rows = []
    for g in groups:
        g_df = base.copy()
        g_df[group_col] = g
        g_rate = float(np.asarray(result.predict(g_df)).mean())
        rows.append({
            "group": g,
            "standardized_rate": g_rate,
            "reference_rate": ref_rate,
            "srr": g_rate / ref_rate if ref_rate > 0 else np.nan,
            "fit_method": getattr(result, "method", "statsmodels-logit"),
        })
    return pd.DataFrame(rows)


def bootstrap_cluster(df: pd.DataFrame, cluster_col: str, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    clusters = np.array(df[cluster_col].dropna().unique())
    grouped = {k: v for k, v in df.groupby(cluster_col, sort=False)}
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for i, cid in enumerate(sampled):
            block = grouped[cid].copy()
            block[cluster_col] = f"{cid}__boot{i}"
            parts.append(block)
        yield pd.concat(parts, ignore_index=True)


def wald_or(beta: float, se: float) -> tuple[float, float, float]:
    lo = beta - 1.96 * se
    hi = beta + 1.96 * se
    return float(np.exp(beta)), float(np.exp(lo)), float(np.exp(hi))
