#!/usr/bin/env python3
"""
Sec. 4.4, Table 3: variable selection on TCGA-KIRC survival data.

Methods
-------
vd_afs_cox    the proposed selector: T-Rex with virtual dummies and AFS on the
              Cox partial likelihood, score refit after every step
vd_afs_ols    the misspecified linear model: Gaussian AFS on centered log(time),
              event indicator discarded
bh_cox        marginal Cox score test per gene + Benjamini-Hochberg
by_cox        marginal Cox score test per gene + Benjamini-Yekutieli
cox_lasso     ell_1 Cox regression (scikit-survival), penalty by cross-validated
              C-index at the one-standard-error rule
cox_knockoff  model-X knockoffs with a Cox Lasso statistic, see cox_knockoff.py

The defaults are the run behind results/sec4_4_kirc/ (alpha = 0.1, K = 20,
L = 10p, rho = 0.1, T_max = 50). From the repository root:

    python experiments/sec4_4_kirc/kirc_data.py
    python experiments/sec4_4_kirc/kirc_selection.py --threads 8

writes sweep_<tag>.csv and diag_<tag>.csv to results/sec4_4_kirc/rerun/;
make_table3.py condenses them.
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

# NumPy >= 2 on macOS links Apple's Accelerate BLAS, whose matrix products can
# leave stale floating-point flags behind. NumPy then reports "divide by zero /
# overflow / invalid value encountered in matmul" although the product is
# finite and correct. scikit-learn and scikit-survival trigger it thousands of
# times inside the Cox Lasso. Silence exactly that message, on macOS only; any
# other RuntimeWarning still shows.
if sys.platform == "darwin":
    import warnings

    warnings.filterwarnings(
        "ignore", category=RuntimeWarning,
        message=r"(divide by zero|overflow|invalid value) encountered in matmul")

try:
    import vd_selectors as vd
except ImportError as _e:  # pragma: no cover
    vd = None
    _VD_ERROR = _e

try:
    from sklearn.model_selection import StratifiedKFold
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
except ImportError as _e:  # pragma: no cover
    StratifiedKFold = None
    CoxnetSurvivalAnalysis = None
    concordance_index_censored = None
    Surv = None
    _SKSURV_ERROR = _e

# The knockoff row lives in its own module: its construction never looks at the
# response and costs orders of magnitude more. Missing knockpy is only an error
# when the method is requested.
KO_METHOD_NAMES = ("cox_knockoff",)
try:
    from cox_knockoff import KO_LABELS, KO_SELECTORS
except ImportError as _e:  # pragma: no cover
    KO_SELECTORS, KO_LABELS = {}, {}
    _KO_ERROR = _e


# ============================================================================
# Generic result container
# ============================================================================


@dataclass
class SelectionResult:
    selected_var: np.ndarray
    method: str
    T_stop: int = 0
    num_dummies: int = 0
    v_thresh: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Virtual-dummy selectors
# ============================================================================


CALIB = {
    "T": "CalibrateT",
    "L": "CalibrateL",
    "both": "CalibrateBoth",
    "fixed": "FixedTL",
}


def _options(
    alpha,
    seed,
    solver,
    K=20,
    L_factor=5,
    T_stop=40,
    rho=0.8,
    n_threads=1,
    max_stale_strides=0,
    stride_width=0,
    verbose=False,
    calib="T",
):
    if vd is None:
        raise ImportError(
            "vd_selectors cannot be imported. In the repository root run "
            f"`python3.11 -m pip install -e .`. Original error: {_VD_ERROR}"
        )
    o = vd.TRexOptions()
    o.tFDR = float(alpha)
    o.K = int(K)
    o.L_factor = int(L_factor)
    o.T_stop = int(T_stop)
    o.rho = float(rho)
    o.calib = getattr(vd.CalibMode, CALIB[calib])
    o.dummy_law = vd.VDDummyLaw.Spherical
    o.solver = solver
    o.n_threads = int(n_threads)
    o.seed = int(seed)
    o.verbose = bool(verbose)
    if max_stale_strides:
        o.max_stale_strides = int(max_stale_strides)
    if stride_width:
        o.stride_width = int(stride_width)
    return o


def _finish_vd(o, X, y, method):
    X = np.asfortranarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    raw = vd.TRexSelector(o).run(X, y)  #type: ignore
    sel = np.sort(np.asarray(raw.selected_var, dtype=int))
    return SelectionResult(
        selected_var=sel,
        method=method,
        T_stop=int(raw.T_stop),
        num_dummies=int(raw.num_dummies),
        v_thresh=float(raw.v_thresh),
    )


def vd_afs_cox(X, time, event, alpha=0.1, seed=0, efron=False, **kw):
    o = _options(alpha, seed, vd.SolverType.AFS_Cox if vd else None, **kw)  #type: ignore
    o.cox_status = np.ascontiguousarray(event, dtype=np.float64)  #type: ignore
    o.cox_efron = bool(efron)  #type: ignore
    y = np.asarray(time, dtype=np.float64)
    if np.any(y <= 0):
        raise ValueError("All survival times must be strictly positive.")
    return _finish_vd(o, X, y, "vd_afs_cox")


def vd_afs_ols(X, time, event=None, alpha=0.1, seed=0, **kw):
    kw.pop("efron", None)
    o = _options(alpha, seed, vd.SolverType.AFS if vd else None, **kw)
    y = np.log(np.asarray(time, dtype=np.float64))
    y = y - y.mean()
    return _finish_vd(o, X, y, "vd_afs_ols")


# ============================================================================
# Correct marginal Cox score test with Breslow ties
# ============================================================================


def univariate_cox_z(X, time, event):
    """
    Marginal Cox score z-statistics at beta=0, using Breslow ties.

    All events at one observed time use the same complete risk set. This fixes
    the common error of updating the risk set within a tied event group.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]

    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    if X.shape[0] != len(time) or len(time) != len(event):
        raise ValueError("X, time, and event have incompatible lengths.")

    order = np.argsort(time, kind="mergesort")
    t = time[order]
    d = event[order]
    Xs = X[order]
    n, p = Xs.shape

    # For ascending time, rows start:n form the risk set at t[start].
    risk_sum = np.cumsum(Xs[::-1], axis=0)[::-1]
    risk_sum_sq = np.cumsum((Xs * Xs)[::-1], axis=0)[::-1]

    _, starts, counts = np.unique(t, return_index=True, return_counts=True)
    score = np.zeros(p, dtype=np.float64)
    information = np.zeros(p, dtype=np.float64)

    for start, count in zip(starts, counts):
        stop = start + count
        n_events = float(d[start:stop].sum())
        if n_events == 0.0:
            continue

        n_at_risk = float(n - start)
        mean_risk = risk_sum[start] / n_at_risk
        var_risk = risk_sum_sq[start] / n_at_risk - mean_risk * mean_risk
        var_risk = np.maximum(var_risk, 0.0)

        observed = np.sum(d[start:stop, None] * Xs[start:stop], axis=0)
        score += observed - n_events * mean_risk
        information += n_events * var_risk

    return score / np.sqrt(np.maximum(information, 1e-300))


def two_sided_normal_p(z):
    """Two-sided standard-normal p-values without a SciPy dependency."""
    z = np.asarray(z, dtype=np.float64)
    flat = np.fromiter(
        (math.erfc(abs(float(v)) / math.sqrt(2.0)) for v in z.ravel()),
        dtype=np.float64,
        count=z.size,
    )
    return flat.reshape(z.shape)


def benjamini_hochberg(p_values, q, dependency="pos"):
    """
    Linear step-up rejection mask and adjusted p-values.

    dependency="pos"
        Benjamini-Hochberg. Valid for independent p-values and under positive
        regression dependence (PRDS).
    dependency="any"
        Benjamini-Yekutieli. The threshold is deflated by the harmonic factor
        c(m) = sum_{i<=m} 1/i, which makes the procedure valid under arbitrary
        dependence. On p ~ 2e4 this costs roughly a factor c(m) ~ 10.4 in the
        effective level, so it is strictly more conservative than BH.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    m = p_values.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)

    if dependency == "pos":
        c_m = 1.0
    elif dependency == "any":
        c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    else:
        raise ValueError("dependency must be 'pos' (BH) or 'any' (BY).")

    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    thresholds = float(q) * np.arange(1, m + 1) / (m * c_m)
    passing = np.flatnonzero(ranked <= thresholds)

    reject = np.zeros(m, dtype=bool)
    if passing.size:
        k = int(passing[-1])
        reject[order[: k + 1]] = True

    adjusted_ranked = ranked * m * c_m / np.arange(1, m + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_ranked
    return reject, adjusted


def storey_pi0(p_values, lam=0.5):
    """
    Storey's estimate of the null proportion, pi0 = #{p_i > lam} / ((1-lam) m).

    Reported as a diagnostic only. It is not used for selection, but it turns
    "most of the transcriptome is marginally associated" into a number.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    return float(min(1.0, np.mean(p > lam) / (1.0 - lam)))


def _marginal_cox_fdr(X, time, event, alpha, dependency, method):
    z = univariate_cox_z(X, time, event)
    p = two_sided_normal_p(z)
    reject, p_adj = benjamini_hochberg(p, alpha, dependency=dependency)
    return SelectionResult(
        selected_var=np.asarray(np.flatnonzero(reject), dtype=int),
        method=method,
        extra={
            "q": float(alpha),
            "z": z,
            "p_value": p,
            "p_adjusted": p_adj,
            "min_p": float(np.min(p)) if p.size else float("nan"),
            "pi0_storey": storey_pi0(p),
            "dependency": dependency,
        },
    )


def bh_cox(X, time, event, alpha=0.1, seed=0, **kw):
    """Marginal Cox score tests followed by Benjamini-Hochberg at q=alpha."""
    del seed, kw
    return _marginal_cox_fdr(X, time, event, alpha, "pos", "bh_cox")


def by_cox(X, time, event, alpha=0.1, seed=0, **kw):
    """
    Marginal Cox score tests followed by Benjamini-Yekutieli at q=alpha.

    Same test statistics as bh_cox, only the step-up threshold changes. This is
    the dependence-robust counterpart: BH needs PRDS, which is not verifiable
    for a gene expression matrix, whereas BY holds under arbitrary dependence.
    """
    del seed, kw
    return _marginal_cox_fdr(X, time, event, alpha, "any", "by_cox")


# ============================================================================
# Cox Lasso with cross-validated lambda
# ============================================================================


def _require_sksurv():
    if CoxnetSurvivalAnalysis is None:
        raise ImportError(
            "cox_lasso requires scikit-survival. Install it with "
            "`python3.11 -m pip install scikit-survival`. "
            f"Original error: {_SKSURV_ERROR}"
        )


def cox_lasso_cv(
    X,
    time,
    event,
    seed=0,
    cv_folds=5,
    rule="1se",
    n_alphas=100,
    alpha_min_ratio=0.01,
    coef_tol=1e-8,
    max_iter=100000,
    tol=1e-7,
    **kw,
):
    """
    L1-penalized Cox model with CV-selected lambda.

    The selected penalty maximizes mean validation Harrell C-index (rule=min)
    or uses the strongest penalty whose mean C-index lies within one standard
    error of the best value (rule=1se). The latter is usually sparser.

    This method does not use the target-FDR alpha. It is intentionally run once
    in alpha_sweep().
    """
    del kw
    _require_sksurv()

    X = np.asarray(X, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event_bool = np.asarray(event, dtype=bool)
    y = Surv.from_arrays(event=event_bool, time=time)  #type: ignore

    class_counts = np.bincount(event_bool.astype(int), minlength=2)
    max_folds = int(class_counts.min())
    n_splits = min(int(cv_folds), max_folds)
    if n_splits < 2:
        raise ValueError("Cox-Lasso CV needs at least two events and two censored cases.")

    # Obtain one common decreasing alpha grid from the full data.
    path_model = CoxnetSurvivalAnalysis(
        l1_ratio=1.0,
        n_alphas=int(n_alphas),
        alpha_min_ratio=float(alpha_min_ratio),  #type: ignore
        normalize=False,
        tol=float(tol),
        max_iter=int(max_iter),
        fit_baseline_model=False,
    )  #type: ignore
    path_model.fit(X, y)
    alpha_grid = np.asarray(path_model.alphas_, dtype=np.float64)
    if alpha_grid.size == 0:
        raise RuntimeError("Coxnet returned an empty penalty path.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed)) #type: ignore
    fold_scores = np.full((n_splits, alpha_grid.size), np.nan, dtype=np.float64)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, event_bool.astype(int))):
        fold_model = CoxnetSurvivalAnalysis(
            l1_ratio=1.0,
            alphas=alpha_grid,
            normalize=False,
            tol=float(tol),
            max_iter=int(max_iter),
            fit_baseline_model=False,
        ) #type:ignore 
        fold_model.fit(X[train_idx], y[train_idx])
        coef_path = np.asarray(fold_model.coef_, dtype=np.float64)
        if coef_path.ndim == 1:
            coef_path = coef_path[:, None]

        for j in range(coef_path.shape[1]):
            risk = X[valid_idx] @ coef_path[:, j]
            try:
                cindex = concordance_index_censored(
                    event_bool[valid_idx], time[valid_idx], risk
                )[0]  #type: ignore
            except Exception:
                cindex = np.nan
            fold_scores[fold, j] = float(cindex)

    mean_c = np.nanmean(fold_scores, axis=0)
    valid_n = np.sum(np.isfinite(fold_scores), axis=0)
    sd_c = np.nanstd(fold_scores, axis=0, ddof=1)
    se_c = sd_c / np.sqrt(np.maximum(valid_n, 1))

    if not np.any(np.isfinite(mean_c)):
        raise RuntimeError("All Cox-Lasso CV scores are missing.")

    best_idx = int(np.nanargmax(mean_c))
    if rule == "min":
        chosen_idx = best_idx
    elif rule == "1se":
        threshold = mean_c[best_idx] - se_c[best_idx]
        eligible = np.flatnonzero(mean_c >= threshold)
        # Choose the strongest penalty among the eligible values, independent
        # of the ordering returned by the library.
        chosen_idx = (
            int(eligible[np.argmax(alpha_grid[eligible])])
            if eligible.size
            else best_idx
        )
    else:
        raise ValueError("lasso rule must be '1se' or 'min'.")

    coef_path_full = np.asarray(path_model.coef_, dtype=np.float64)
    if coef_path_full.ndim == 1:
        coef_path_full = coef_path_full[:, None]
    coef = coef_path_full[:, chosen_idx]
    sel = np.flatnonzero(np.abs(coef) > float(coef_tol))

    return SelectionResult(
        selected_var=np.asarray(sel, dtype=int),
        method="cox_lasso",
        extra={
            "chosen_lambda": float(alpha_grid[chosen_idx]),
            "chosen_index": chosen_idx,
            "best_index": best_idx,
            "cv_cindex": float(mean_c[chosen_idx]),
            "best_cv_cindex": float(mean_c[best_idx]),
            "cv_se": float(se_c[chosen_idx]),
            "rule": rule,
            "coef": coef,
            "alpha_grid": alpha_grid,
            "mean_cindex_path": mean_c,
            "se_cindex_path": se_c,
        },
    )


# ============================================================================
# Complementary-pairs stability selection on the Cox Lasso path
# ============================================================================


def _coxnet_coef_path(X, y, alpha_grid, max_iter, tol):
    """
    Coxnet coefficient path on a fixed penalty grid, robust to non-convergence.

    Coxnet raises on the small-penalty end of the grid when the sub-sample is
    degenerate. We then drop the smallest penalties and retry, which is fine
    here because stability selection only reads the large-penalty end.
    """
    grid = np.asarray(alpha_grid, dtype=np.float64)
    while grid.size >= 2:
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=1.0,
                alphas=grid,
                normalize=False,
                tol=float(tol),
                max_iter=int(max_iter),
                fit_baseline_model=False,
            )  # type: ignore
            model.fit(X, y)
            coef = np.asarray(model.coef_, dtype=np.float64)
            return coef[:, None] if coef.ndim == 1 else coef
        except Exception:  # noqa: BLE001
            grid = grid[: max(2, grid.size // 2)]
    return None


# ============================================================================
# Selector registry and dispatch
# ============================================================================


SELECTORS: dict[str, Callable[..., SelectionResult]] = {
    "vd_afs_cox": vd_afs_cox,
    "vd_afs_ols": vd_afs_ols,
    "bh_cox": bh_cox,
    "by_cox": by_cox,
    "cox_lasso": cox_lasso_cv,
    **KO_SELECTORS,
}

LABELS = {
    "vd_afs_cox": "VD-AFS-Cox",
    "vd_afs_ols": "VD-AFS-Gaussian on log(time)",
    "bh_cox": "Marginal Cox + BH",
    "by_cox": "Marginal Cox + BY",
    "cox_lasso": "Cox Lasso, CV",
    **KO_LABELS,
}

# The Cox Lasso has no target FDR, so it runs once instead of once per alpha.
SINGLE_RUN_METHODS = {"cox_lasso"}


def run_selector(method, X, time, event, alpha, seed, vd_kw, lasso_kw, ko_kw=None):
    if method == "cox_lasso":
        return cox_lasso_cv(X, time, event, seed=seed, **lasso_kw)
    if method in KO_METHOD_NAMES:
        return SELECTORS[method](X, time, event, alpha=alpha, seed=seed,
                                 **(ko_kw or {}))
    if method in ("bh_cox", "by_cox"):
        return SELECTORS[method](X, time, event, alpha=alpha, seed=seed)
    return SELECTORS[method](X, time, event, alpha=alpha, seed=seed, **vd_kw)


# ============================================================================
# Helpers
# ============================================================================


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024**2) if sys.platform == "darwin" else r / 1024


# ============================================================================
# Runs and summaries
# ============================================================================


def _alpha_label(alpha):
    return "CV" if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)) else f"{alpha:g}"


def alpha_sweep(methods, X, time, event, genes, alphas, vd_kw, lasso_kw, seed,
                ko_kw=None):
    """Run all requested methods and print one compact line per fit."""
    rows, picks = [], {}
    for method in methods:
        method_alphas = [None] if method in SINGLE_RUN_METHODS else alphas

        for alpha in method_alphas:
            t0 = _time.perf_counter()
            result = run_selector(
                method,
                X,
                time,
                event,
                alpha=alpha,
                seed=seed,
                vd_kw=vd_kw,
                lasso_kw=lasso_kw,
                ko_kw=ko_kw,
            )
            dt = _time.perf_counter() - t0
            sel = np.sort(np.asarray(result.selected_var, dtype=int))
            names = [str(g) for g in np.asarray(genes)[sel]]
            picks[(method, alpha)] = (sel, names, result)

            row = dict(
                method=method,
                alpha=np.nan if alpha is None else float(alpha),
                tuning="CV" if alpha is None else "target_fdr",
                n_selected=len(sel),
                seconds=round(dt, 3),
                peak_rss_mb=round(peak_rss_mb(), 1),
                T_stop=int(result.T_stop),
                L=int(result.num_dummies),
                v_thresh=float(result.v_thresh),
                genes=";".join(names),
            )
            if method == "cox_lasso":
                row.update(
                    chosen_lambda=result.extra["chosen_lambda"],
                    cv_cindex=result.extra["cv_cindex"],
                    cv_rule=result.extra["rule"],
                )
            elif method in ("bh_cox", "by_cox"):
                row.update(
                    min_p=result.extra["min_p"],
                    pi0_storey=result.extra["pi0_storey"],
                )
            elif method in KO_METHOD_NAMES:
                e = result.extra
                row.update(
                    ko_type=e["ko_type"],
                    ko_tau=e["ko_tau"],
                    ko_lambda=e["ko_lambda"],
                    ko_nonzero=e["ko_nonzero"],
                    ko_s_min=e.get("ko_s_min"),
                    ko_s_max=e.get("ko_s_max"),
                    ko_cached=e.get("ko_cached"),
                )
            rows.append(row)

            shown = ", ".join(names[:8])
            if len(names) > 8:
                shown += f", ... (+{len(names) - 8})"
            shown = shown or "(empty)"
            level = "CV" if alpha is None else f"q={alpha:g}"
            print(
                f"{LABELS.get(method, method):<16} {level:>7} "
                f"{len(sel):>7} genes  {dt:>7.2f}s  {shown}"
            )

    return pd.DataFrame(rows), picks


def diagnose(picks, X, time, event, genes, verbose=True):
    """Descriptive, post-selection marginal diagnostics; not valid inference."""
    X = np.asarray(X, dtype=np.float64)
    y = np.log(np.asarray(time, dtype=np.float64))
    y = y - y.mean()
    y /= max(np.linalg.norm(y), 1e-300)
    d = np.asarray(event, dtype=np.float64)
    d = d - d.mean()
    d /= max(np.linalg.norm(d), 1e-300)

    rows = []
    for (method, alpha), (sel, names, _) in picks.items():
        if len(sel) == 0:
            continue
        z = univariate_cox_z(X[:, sel], time, event)
        r_t = X[:, sel].T @ y
        r_event = X[:, sel].T @ d
        for j, gene in enumerate(names):
            rows.append(
                dict(
                    method=method,
                    alpha=np.nan if alpha is None else alpha,
                    gene=gene,
                    z_cox=float(z[j]),
                    r_logt=float(r_t[j]),
                    r_event=float(r_event[j]),
                )
            )
    df = pd.DataFrame(rows)
    if df.empty or not verbose:
        return df

    print("\n" + "=" * 72)
    print("Descriptive post-selection diagnostics")
    print("=" * 72)
    print(f"  {'method':<14} {'q/tune':>7} {'gene':<14} {'z_cox':>8} {'r_logt':>8} {'r_event':>9}")
    for _, row in df.iterrows():
        alpha = None if pd.isna(row.alpha) else float(row.alpha)
        print(
            f"  {row.method:<14} {_alpha_label(alpha):>7} {row.gene:<14} "
            f"{row.z_cox:>8.2f} {row.r_logt:>8.3f} {row.r_event:>9.3f}"
        )

    print("\n  These z values use the same data as selection and are descriptive only.")
    for method in df.method.unique():
        unique = df[df.method == method].drop_duplicates(subset=["gene"])
        print(
            f"  {method}: {(unique.z_cox.abs() > 1.96).sum()} of {len(unique)} "
            "unique genes have |z_cox| > 1.96"
        )
    return df


# ============================================================================
# Null and split-half controls
# ============================================================================


# ============================================================================
# Command line
# ============================================================================


def build_parser():
    ap = argparse.ArgumentParser(
        description="Table 3: VD-AFS-Cox and the benchmark rows on TCGA-KIRC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--npz", default="data/kirc_OS.npz")
    ap.add_argument("--cohort", default="KIRC")
    ap.add_argument("--methods", default="vd_afs_cox,vd_afs_ols,bh_cox,by_cox,cox_lasso",
                    help="comma-separated list from " + ", ".join(
                        ["vd_afs_cox", "vd_afs_ols", "bh_cox", "by_cox", "cox_lasso",
                         "cox_knockoff"]))
    ap.add_argument("--alphas", default="0.1")
    ap.add_argument("--out-dir", default="results/sec4_4_kirc/rerun")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--verbose-output", action="store_true",
                    help="print the diagnostics of the selected genes")

    g = ap.add_argument_group("T-Rex / virtual dummies")
    g.add_argument("--K", type=int, default=20)
    g.add_argument("--L", type=int, default=10, help="L factor, so total L = factor * p")
    g.add_argument("--T-stop", type=int, default=50)
    g.add_argument("--rho", type=float, default=0.1)
    g.add_argument("--threads", type=int, default=1)

    g = ap.add_argument_group("Cox Lasso")
    g.add_argument("--lasso-folds", type=int, default=5)
    g.add_argument("--lasso-rule", choices=["1se", "min"], default="1se")
    g.add_argument("--lasso-n-alphas", type=int, default=100)
    g.add_argument("--lasso-alpha-min-ratio", type=float, default=0.01)
    g.add_argument("--lasso-coef-tol", type=float, default=1e-8)
    g.add_argument("--lasso-max-iter", type=int, default=100000)
    g.add_argument("--lasso-tol", type=float, default=1e-7)

    g = ap.add_argument_group("Knockoffs")
    g.add_argument("--ko-method", default=None,
                   choices=["mvr", "maxent", "sdp", "equicorrelated"],
                   help="S-matrix construction, unset is knockpy's default (MVR)")
    g.add_argument("--ko-shrinkage", default="ledoitwolf",
                   choices=["ledoitwolf", "graphicallasso", "none"])
    g.add_argument("--ko-max-block", type=int, default=1000)
    g.add_argument("--ko-processes", type=int, default=1)
    g.add_argument("--ko-lasso-rule", choices=["min", "1se"], default="min")
    g.add_argument("--ko-cache-dir", default=None, help="where X_tilde is cached")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    alphas = [float(a) for a in args.alphas.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in SELECTORS]
    if unknown:
        if any(m in KO_METHOD_NAMES for m in unknown):
            raise SystemExit(f"cox_knockoff needs knockpy. Original error: {_KO_ERROR}")
        raise SystemExit(f"Unknown methods {unknown}. Allowed: {list(SELECTORS)}")

    vd_kw = dict(K=args.K, L_factor=args.L, T_stop=args.T_stop, rho=args.rho,
                 n_threads=args.threads)
    lasso_kw = dict(
        cv_folds=args.lasso_folds, rule=args.lasso_rule,
        n_alphas=args.lasso_n_alphas, alpha_min_ratio=args.lasso_alpha_min_ratio,
        coef_tol=args.lasso_coef_tol, max_iter=args.lasso_max_iter, tol=args.lasso_tol,
    )
    ko_kw = dict(
        ko_method=args.ko_method, ko_shrinkage=args.ko_shrinkage,
        ko_max_block=args.ko_max_block, ko_num_processes=args.ko_processes,
        ko_lasso_rule=args.ko_lasso_rule, ko_cache_dir=args.ko_cache_dir,
        ko_cv=args.lasso_folds, ko_n_alphas=args.lasso_n_alphas,
        ko_alpha_min_ratio=args.lasso_alpha_min_ratio, ko_coef_tol=args.lasso_coef_tol,
        ko_max_iter=args.lasso_max_iter, ko_tol=args.lasso_tol,
    )

    from kirc_data import CohortData, check_h

    npz = Path(args.npz)
    if not npz.exists():
        raise SystemExit(f"File not found: {npz}. Build it with "
                         "experiments/sec4_4_kirc/kirc_data.py")
    data = CohortData.load(npz)
    X, time, event, genes = data.X, data.time, data.event, data.genes
    cohort = args.cohort.upper()
    tag = f"{cohort}_n{data.n}_p{data.p}"
    print(f"[data] {cohort}: {data.summary()}")
    if cohort == "KIRC" and (data.n, data.p, int(data.event.sum())) != (531, 19620, 175):
        print("[warn] this is not the paper cohort (n=531, p=19620, 175 "
              "events). Rebuild it with experiments/sec4_4_kirc/kirc_data.py")
    check_h(X)

    n, p = X.shape
    print("\n" + "=" * 72)
    print(f"n={n} p={p} Events={int(np.sum(event))} L={vd_kw['L_factor'] * p} "
          f"K={args.K} rho={args.rho} methods={', '.join(methods)}")
    print("=" * 72)

    # The knockoff cache is keyed by this tag, so a rerun reuses X_tilde.
    sweep, picks = alpha_sweep(methods, X, time, event, genes, alphas, vd_kw,
                               lasso_kw, args.seed, ko_kw=dict(ko_kw, tag=tag))
    sweep.to_csv(out / f"sweep_{tag}.csv", index=False)

    # Marginal diagnostics for the genes the VD selectors pick.
    vd_picks = {k: v for k, v in picks.items() if k[0].startswith("vd_")}
    diag = diagnose(vd_picks, X, time, event, genes, verbose=args.verbose_output)
    if not diag.empty:
        diag.to_csv(out / f"diag_{tag}.csv", index=False)

    meta = dict(tag=tag, n=int(n), p=int(p), events=int(np.sum(event)),
                methods=methods, alphas=alphas, peak_rss_mb=round(peak_rss_mb(), 1),
                vd_options=vd_kw, lasso_options=lasso_kw,
                knockoff_options=ko_kw if "cox_knockoff" in methods else None)
    (out / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nResults saved in: {out.resolve()}")


if __name__ == "__main__":
    main()

