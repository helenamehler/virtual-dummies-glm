#!/usr/bin/env python3
"""
Model-X knockoff row of Table 3 (Sec. 4.4): CoxKnockoff (Li, Yu and Zhao, Stat
2023) on all genes.

The construction depends on X only and is knockpy's (Ledoit-Wolf covariance,
S-matrix, GaussianSampler); at p = 19 620 it is the expensive half, so X_tilde
is cached. The statistic W_j = |beta_j| - |beta_{j+p}| comes from the Cox Lasso
of kirc_selection.py on [X, X_tilde], and the knockoff+ threshold gives the
selection.

    # precompute and cache X_tilde for one design
    python cox_knockoff.py generate data/kirc_OS.npz <cache_dir> [cfg_json]

kirc_selection.py runs the statistic as method ``cox_knockoff``.
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

__all__ = [
    "DEFAULT_CFG",
    "sample_knockoffs",
    "knockoff_cache_path",
    "build_or_load_knockoffs",
    "cox_lcd",
    "knockoff_threshold",
    "cox_knockoff",
    "KO_SELECTORS",
    "KO_LABELS",
]

DEFAULT_CFG = {
    # --- knockpy construction --------------------------------------------
    "ko_shrinkage": "ledoitwolf",    # utilities.estimate_covariance
    "ko_cov_tol": 1e-4,
    "ko_method": None,               # None -> knockpy's default, MVR
    "ko_how_approx": "blockdiag",
    "ko_max_block": 1000,
    "ko_num_processes": 1,
    "ko_cache_dir": None,
    # --- the Cox statistic -------------------------------------------------
    "ko_lasso_rule": "min",          # lambda maximising the CV C-index
    "ko_cv": 5,
    "ko_n_alphas": 100,
    "ko_alpha_min_ratio": 0.01,
    "ko_coef_tol": 1e-8,
    "ko_max_iter": 100000,
    "ko_tol": 1e-7,
    # --- the filter ---------------------------------------------------------
    "ko_offset": 1,                  # 1 = knockoff+ (FDR control)
    "ko_verbose": True,
}


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _cfg(cfg, key):
    return cfg.get(key, DEFAULT_CFG[key])


# =====================================================================
# 1. Knockoff generation, delegated to knockpy
# =====================================================================


def sample_knockoffs(Xc, cfg, seed=0, verbose=True):
    """Second-order model-X Gaussian knockoffs, entirely via knockpy.

    Parameters
    ----------
    Xc : (n, p) ndarray
        Column-centred, ell_2-normalised design, i.e. what ``CohortData.load``
        hands to every other selector in ``kirc_selection.py``.
    cfg : dict
        Reads ``ko_shrinkage``, ``ko_cov_tol``, ``ko_method``,
        ``ko_how_approx``, ``ko_max_block``, ``ko_num_processes``.

    Returns
    -------
    Xk : (n, p) ndarray on the same column scale as ``Xc``.
    info : dict

    Notes
    -----
    knockpy works in the unit-variance scale, so the design goes in as
    ``Z = sqrt(n) Xc`` and the result is divided back. No per-column
    renormalisation afterwards: that would break pairwise exchangeability, and
    the Cox-Lasso needs originals and knockoffs on one common scale anyway.

    ``Sigma`` is estimated, never assumed. On an i.i.d. design Ledoit-Wolf lands
    close to the identity by itself, which is the right answer arrived at the
    honest way rather than put in by hand.
    """
    from knockpy import smatrix, utilities
    from knockpy.knockoffs import GaussianSampler

    Xc = np.asarray(Xc, dtype=np.float64)
    n, p = Xc.shape
    t0 = time.perf_counter()
    Z = np.sqrt(n) * Xc

    shrinkage = _cfg(cfg, "ko_shrinkage")
    Sigma, invSigma = utilities.estimate_covariance(
        Z, float(_cfg(cfg, "ko_cov_tol")), shrinkage
    )
    t_cov = time.perf_counter() - t0

    method = _cfg(cfg, "ko_method")
    S = smatrix.compute_smatrix(
        Sigma,
        method=method,
        how_approx=_cfg(cfg, "ko_how_approx"),
        max_block=int(_cfg(cfg, "ko_max_block")),
        num_processes=int(_cfg(cfg, "ko_num_processes")),
    )
    t_s = time.perf_counter() - t0 - t_cov

    # GaussianSampler draws from numpy's global stream
    np.random.seed(int(seed))
    sampler = GaussianSampler(
        Z, mu=np.zeros(p), Sigma=Sigma, invSigma=invSigma, S=S
    )
    Xk = sampler.sample_knockoffs() / np.sqrt(n)

    s_diag = np.diag(S)
    info = {
        "ko_backend": "knockpy",
        "ko_shrinkage": str(shrinkage),
        "ko_method": "mvr (knockpy default)" if method is None else str(method),
        "ko_how_approx": str(_cfg(cfg, "ko_how_approx")),
        "ko_max_block": int(_cfg(cfg, "ko_max_block")),
        "ko_s_min": float(s_diag.min()),
        "ko_s_max": float(s_diag.max()),
        "ko_s_mean": float(s_diag.mean()),
        "ko_n": int(n),
        "ko_p": int(p),
        "ko_cov_seconds": round(t_cov, 2),
        "ko_smatrix_seconds": round(t_s, 2),
        "ko_gen_seconds": round(time.perf_counter() - t0, 2),
    }
    if verbose:
        _log(
            f"  knockpy: n={n} p={p} shrinkage={shrinkage} "
            f"method={info['ko_method']} s in "
            f"[{info['ko_s_min']:.4g}, {info['ko_s_max']:.4g}] "
            f"(cov {t_cov:.1f}s, S {t_s:.1f}s, "
            f"total {info['ko_gen_seconds']:.1f}s)"
        )
    return Xk, info


def knockoff_cache_path(cache_dir, tag, cfg, seed=0):
    method = _cfg(cfg, "ko_method") or "mvr"
    shrink = _cfg(cfg, "ko_shrinkage")
    return Path(cache_dir) / f"knockoffs_{tag}_{method}_{shrink}_s{int(seed)}.npy"


def build_or_load_knockoffs(Xc, cfg, tag=None, seed=0):
    """Cache-aware wrapper, keyed by tag, method, shrinkage and seed.

    The KIRC design is one fixed matrix, so unlike the HAPNEST replicates the
    cache here amortises over everything: the alpha sweep, both knockoff rows,
    a second statistic, a rerun after a plotting bug. Given that the
    construction is hours and the statistic is minutes, that is the whole
    difference between iterating and waiting.
    """
    Xc = np.asarray(Xc, dtype=np.float64)
    cache_dir = _cfg(cfg, "ko_cache_dir")
    path = None
    if cache_dir and tag is not None:
        path = knockoff_cache_path(cache_dir, tag, cfg, seed)
        if path.exists():
            Xk = np.load(path)
            if Xk.shape == Xc.shape:
                _log(f"  knockpy: loaded cache {path.name}")
                return np.asarray(Xk, dtype=np.float64), {
                    "ko_backend": "knockpy",
                    "ko_cached": True,
                    "ko_n": int(Xc.shape[0]),
                    "ko_p": int(Xc.shape[1]),
                }
            _log(
                f"  knockpy: cache {path.name} has shape {Xk.shape}, "
                f"expected {Xc.shape}; regenerating"
            )

    Xk, info = sample_knockoffs(
        Xc, cfg, seed=seed, verbose=bool(_cfg(cfg, "ko_verbose"))
    )
    info["ko_cached"] = False
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:  # np.save would re-append ".npy"
            np.save(fh, Xk)
        os.replace(tmp, path)
        _log(f"  knockpy: cached to {path}")
    return Xk, info


# =====================================================================
# 2. The Cox feature statistic
# =====================================================================


def _lasso_kw(cfg):
    return dict(
        cv_folds=int(_cfg(cfg, "ko_cv")),
        rule=str(_cfg(cfg, "ko_lasso_rule")),
        n_alphas=int(_cfg(cfg, "ko_n_alphas")),
        alpha_min_ratio=float(_cfg(cfg, "ko_alpha_min_ratio")),
        coef_tol=float(_cfg(cfg, "ko_coef_tol")),
        max_iter=int(_cfg(cfg, "ko_max_iter")),
        tol=float(_cfg(cfg, "ko_tol")),
    )


def cox_lcd(Xc, Xk, time_, event, cfg, seed=0):
    """Cox-Lasso coefficient-difference statistic.

        W_j = |beta_j(lambda)| - |beta_{j+p}(lambda)|

    with ``beta`` the penalised partial-likelihood fit on ``[X X_tilde]`` at a
    cross-validated ``lambda``. This is the LCD of CoxKnockoff (Li, Yu and
    Zhao 2023) and the Cox case of ``knockoff::stat.glmnet_coefdiff``.

    The fit is ``kirc_selection.cox_lasso_cv``, i.e. literally the ``cox_lasso``
    row of the table, so the two rows share their solver, their penalty path and
    their CV rule and differ only in the error criterion. ``ko_lasso_rule``
    defaults to ``"min"`` rather than the ``"1se"`` of the ``cox_lasso`` row:
    knockoffs do their own thresholding afterwards, so an extra standard error of
    shrinkage in the statistic only costs power.

    Two knockpy conventions are kept. The ``2p`` columns are randomly permuted
    before the fit, which ``knockoff_stats.fit_lasso`` does so the solver cannot
    break ties by column position, and the antisymmetric combination is
    ``knockoff_stats.combine_Z_stats(Z, antisym="cd")`` rather than a hand-written
    subtraction.
    """
    from knockpy import knockoff_stats as ks
    from knockpy import utilities

    from kirc_selection import _coxnet_coef_path, cox_lasso_cv

    Xc = np.asarray(Xc, dtype=np.float64)
    Xk = np.asarray(Xk, dtype=np.float64)
    n, p = Xc.shape
    if Xk.shape != Xc.shape:
        raise ValueError(f"X_tilde has shape {Xk.shape}, expected {Xc.shape}")

    # knockpy's own trick: randomise the column order so the solver cannot break
    # ties by position, then undo the permutation on the coefficients
    np.random.seed(int(seed))
    inds, rev_inds = utilities.random_permutation_inds(2 * p)
    Xaug = np.hstack([Xc, Xk])[:, inds]

    t0 = time.perf_counter()
    res = cox_lasso_cv(Xaug, time_, event, seed=int(seed), **_lasso_kw(cfg))
    coef = np.asarray(res.extra["coef"], dtype=np.float64)
    coef_tol = float(_cfg(cfg, "ko_coef_tol"))
    n_bumps = 0

    # The CV can land on a penalty at which every coefficient is zero. Then Z is
    # identically zero, W is identically zero, and the row reports "0 genes"
    # without saying anything about knockoffs. Walk down the path to the
    # strongest penalty that admits a nonzero coefficient. This stays a valid
    # feature statistic: the rule reads the augmented matrix only through
    # quantities that are invariant under swapping a column with its knockoff.
    if not np.any(np.abs(coef) > coef_tol):
        from sksurv.util import Surv

        grid = np.asarray(res.extra["alpha_grid"], dtype=np.float64)
        y = Surv.from_arrays(
            event=np.asarray(event, dtype=bool),
            time=np.asarray(time_, dtype=np.float64),
        )
        path = _coxnet_coef_path(
            Xaug, y, grid, int(_cfg(cfg, "ko_max_iter")), float(_cfg(cfg, "ko_tol"))
        )
        if path is not None:
            nz = np.flatnonzero((np.abs(path) > coef_tol).any(axis=0))
            if nz.size:
                order = np.argsort(-grid[: path.shape[1]][nz])  # strongest first
                col = int(nz[order[0]])
                coef = path[:, col]
                n_bumps = 1
                res.extra["chosen_lambda"] = float(grid[col])

    Z = np.abs(coef)[rev_inds]  # undo the permutation
    W = ks.combine_Z_stats(Z, antisym="cd")

    info = {
        "ko_stat": "cox_lcd",
        "ko_lasso_rule": str(_cfg(cfg, "ko_lasso_rule")),
        "ko_lambda": float(res.extra["chosen_lambda"]),
        "ko_cv_cindex": float(res.extra["cv_cindex"]),
        "ko_cv_folds": int(_cfg(cfg, "ko_cv")),
        "ko_bumps": int(n_bumps),
        "ko_nonzero": int(np.sum(Z > coef_tol)),
        "ko_fit_seconds": round(time.perf_counter() - t0, 2),
    }
    return W, info


def knockoff_threshold(W, q, offset=1):
    """knockpy's ``data_dependent_threshhold``.

    ``offset=1`` is knockoff+, which controls the FDR itself; ``offset=0`` is the
    plain knockoff threshold, which controls the modified FDR. Returns ``inf``
    when nothing can be rejected.
    """
    from knockpy import knockoff_stats as ks

    t = float(
        ks.data_dependent_threshhold(
            np.asarray(W, dtype=float), fdr=float(q), offset=int(offset)
        )
    )
    return np.inf if not np.isfinite(t) else t


def _select(W, alpha, cfg):
    tau = knockoff_threshold(W, alpha, offset=int(_cfg(cfg, "ko_offset")))
    if not np.isfinite(tau):
        return np.array([], dtype=int), float("inf")
    return np.where(np.asarray(W) >= tau)[0].astype(int), float(tau)


def _result(sel, method, extra):
    """A ``kirc_selection.SelectionResult``, imported late.

    Late on purpose: ``kirc_selection`` imports this module at the top, so a
    module-level import here would close the cycle. By the time any selector
    runs, that module is fully imported.
    """
    from kirc_selection import SelectionResult

    return SelectionResult(
        selected_var=np.sort(np.asarray(sel, dtype=int)),
        method=method,
        extra=extra,
    )


# =====================================================================
# 3. The selector
# =====================================================================


def cox_knockoff(X, time_, event, alpha=0.1, seed=0, tag=None, **cfg):
    """Model-X knockoffs on all p genes with the Cox LCD statistic.

    CoxKnockoff (Li, Yu and Zhao 2023) as published: second-order Gaussian
    knockoffs, lasso coefficient difference from the penalised partial
    likelihood, knockoff+ threshold at the target FDR.
    """
    X = np.asarray(X, dtype=np.float64)
    Xk, ko_info = build_or_load_knockoffs(X, cfg, tag=tag, seed=seed)
    W, w_info = cox_lcd(X, Xk, time_, event, cfg, seed=seed)
    sel, tau = _select(W, alpha, cfg)
    return _result(
        sel,
        "cox_knockoff",
        {
            **ko_info,
            **w_info,
            "ko_type": "model-X",
            "ko_tau": tau,
            "ko_offset": int(_cfg(cfg, "ko_offset")),
            "W": W,
        },
    )


KO_SELECTORS = {"cox_knockoff": cox_knockoff}

KO_LABELS = {"cox_knockoff": "Model-X knockoffs, Cox-Lasso coefficient difference"}


# =====================================================================
# 4. CLI
# =====================================================================


def _cli_generate(argv):
    """Precompute and cache X_tilde for one npz design.

    The cache key has to match the one ``kirc_selection.main`` builds, which
    is ``f"{cohort}_n{n}_p{p}"``, otherwise the run reads a miss and pays for the
    construction a second time. Pass ``cohort`` (and ``seed``) in the config JSON
    exactly as the run will pass ``--cohort`` and ``--seed``.
    """
    if not argv:
        raise SystemExit("usage: cox_knockoff.py generate <npz> [cache_dir] [cfg_json]")
    npz = Path(argv[0])
    cache_dir = argv[1] if len(argv) > 1 else "ko_cache"
    cfg = json.loads(argv[2]) if len(argv) > 2 else {}
    cfg.setdefault("ko_cache_dir", cache_dir)
    seed = int(cfg.pop("seed", 0))
    cohort = str(cfg.pop("cohort", "") or npz.stem).upper()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kirc_data import CohortData

    data = CohortData.load(npz)
    tag = cfg.pop("tag", None) or f"{cohort}_n{data.n}_p{data.p}"
    _log(f"{tag}: n={data.n} p={data.p} events={int(np.sum(data.event))} seed={seed}")

    Xk, info = build_or_load_knockoffs(data.X, cfg, tag=tag, seed=seed)
    path = knockoff_cache_path(cache_dir, tag, cfg, seed)
    print(json.dumps({"tag": tag, "path": str(path), **info}, indent=1))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "generate":
        _cli_generate(sys.argv[2:])
    else:
        raise SystemExit(f"unknown command {cmd!r}; use generate")


if __name__ == "__main__":
    main()
