#!/usr/bin/env python
"""
Model-X knockoff competitor of Sec. 4.3, with a multinomial feature statistic.

The construction depends on X only and is knockpy's (Ledoit-Wolf covariance,
S-matrix, GaussianSampler); it is the expensive half, so it is cached per
replicate. The statistic W_j = ||beta_j|| - ||beta_{j+p}||, aggregated over the
classes, comes from an ell_1-penalised multinomial logit on [X, X_tilde] with
the penalty chosen by cross-validation. The knockoff+ threshold gives the
selection.

    # precompute and cache the knockoff matrix for one replicate
    python knockoff_multinomial.py generate <data_dir> <run_id> <cache_dir> [cfg_json]

benchmark_worker.py runs the statistic as method ``ko_mx_multinom``.
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), adapted from the binary HAPNEST pipeline of Koka et al. (2026). GPLv3, see NOTICE.

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

__all__ = [
    "sample_knockoffs", "knockoff_threshold_plus", "multinomial_lcd",
    "knockoff_cache_path", "build_or_load_knockoffs",
    "m_ko_mx_multinom", "KO_DISPATCH",
]

DEFAULT_CFG = {
    # --- knockpy construction -------------------------------------------
    "ko_shrinkage": "ledoitwolf",   # utilities.estimate_covariance
    "ko_cov_tol": 1e-4,
    "ko_method": None,              # None -> knockpy's default, MVR
    "ko_how_approx": "blockdiag",
    "ko_max_block": 1000,
    "ko_num_processes": 1,
    "ko_line_search": True,         # knockpy's default; see note in sample_knockoffs
    "ko_seed": 42,
    "ko_cache_dir": None,
    # --- the statistic ---------------------------------------------------
    "ko_fstat": "multinomial_l1",   # or "ovr_l1" (faster, still valid)
    "ko_agg": "l2",                 # aggregation across classes: l2, max, sum
    "ko_n_lambda": 5,
    "ko_c_span": 50.0,
    "ko_cv": 3,
    "ko_max_iter": 200,
    "ko_tol": 1e-3,
    "ko_n_jobs": 1,   # limited by memory, not by cores; see multinomial_lcd
}


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _cfg(cfg, key):
    return cfg.get(key, DEFAULT_CFG[key])


# =====================================================================
# 1. Knockoff generation, delegated to knockpy
# =====================================================================

def sample_knockoffs(Xc, cfg, verbose=True):
    """Second-order model-X Gaussian knockoffs, entirely via knockpy.

    Parameters
    ----------
    Xc : (n, p) ndarray
        Column-centered, ell_2-normalised design, i.e. what the HAPNEST
        preprocessing writes to ``X_std.dat``.
    cfg : dict
        Reads ``ko_shrinkage``, ``ko_cov_tol``, ``ko_method``,
        ``ko_how_approx``, ``ko_max_block``, ``ko_num_processes``, ``ko_seed``.

    Returns
    -------
    Xk : (n, p) ndarray on the same column scale as ``Xc``.
    info : dict

    Notes
    -----
    knockpy works in the unit-variance scale, so the design is passed as
    ``Z = sqrt(n) Xc`` and the result divided back. No per-column
    renormalisation afterwards: that would break pairwise exchangeability.

    ``Sigma`` is estimated from the data, never assumed. On an i.i.d. design
    Ledoit-Wolf lands close to the identity on its own, which is the right
    answer arrived at the honest way rather than put in by hand.
    """
    from knockpy import smatrix, utilities
    from knockpy.knockoffs import GaussianSampler

    Xc = np.asarray(Xc, dtype=np.float64)
    n, p = Xc.shape
    t0 = time.perf_counter()
    Z = np.sqrt(n) * Xc

    # smatrix.divide_computation perturbs Sigma with np.random.randn before the
    # hierarchical clustering, so the block partition comes off numpy's global
    # stream. Seed here as well, or the cached matrix is not reproducible.
    np.random.seed(int(cfg.get("ko_seed", DEFAULT_CFG["ko_seed"])))

    shrinkage = _cfg(cfg, "ko_shrinkage")
    Sigma, invSigma = utilities.estimate_covariance(
        Z, _cfg(cfg, "ko_cov_tol"), shrinkage)
    t_cov = time.perf_counter() - t0

    method = _cfg(cfg, "ko_method")
    line_search = bool(_cfg(cfg, "ko_line_search"))
    S = smatrix.compute_smatrix(
        Sigma,
        method=method,
        how_approx=_cfg(cfg, "ko_how_approx"),
        max_block=int(_cfg(cfg, "ko_max_block")),
        num_processes=int(_cfg(cfg, "ko_num_processes")),
        line_search=line_search,
    )
    t_s = time.perf_counter() - t0 - t_cov

    # GaussianSampler draws from numpy's global stream
    np.random.seed(int(cfg.get("ko_seed", DEFAULT_CFG["ko_seed"])))
    sampler = GaussianSampler(Z, mu=np.zeros(p), Sigma=Sigma,
                              invSigma=invSigma, S=S)
    Xk = sampler.sample_knockoffs() / np.sqrt(n)

    s_diag = np.diag(S)
    info = {
        "ko_backend": "knockpy",
        "ko_shrinkage": shrinkage,
        "ko_method": "mvr (knockpy default)" if method is None else str(method),
        "ko_how_approx": _cfg(cfg, "ko_how_approx"),
        "ko_max_block": int(_cfg(cfg, "ko_max_block")),
        "ko_line_search": line_search,
        "ko_s_min": float(s_diag.min()), "ko_s_max": float(s_diag.max()),
        "ko_s_mean": float(s_diag.mean()),
        "ko_n": n, "ko_p": p,
        "ko_cov_seconds": round(t_cov, 2),
        "ko_smatrix_seconds": round(t_s, 2),
        "ko_gen_seconds": round(time.perf_counter() - t0, 2),
    }
    if verbose:
        _log(f"  knockpy: n={n} p={p} shrinkage={shrinkage} "
             f"method={info['ko_method']} s in "
             f"[{info['ko_s_min']:.4f}, {info['ko_s_max']:.4f}] "
             f"(cov {t_cov:.1f}s, S {t_s:.1f}s, "
             f"total {info['ko_gen_seconds']:.1f}s)")
    return Xk, info


def knockoff_cache_path(cache_dir, run_id, cfg):
    method = _cfg(cfg, "ko_method") or "mvr"
    shrink = _cfg(cfg, "ko_shrinkage")
    seed = int(cfg.get("ko_seed", DEFAULT_CFG["ko_seed"]))
    ls = "" if bool(_cfg(cfg, "ko_line_search")) else "_nols"
    return Path(cache_dir) / (f"run_{run_id}_knockoffs_"
                              f"{method}_{shrink}_s{seed}{ls}.npy")


def build_or_load_knockoffs(Xc, cfg, run_id=None):
    """Cache-aware wrapper, keyed by run_id, method, shrinkage and seed.

    The HAPNEST genotypes differ per replicate (100 independent runs with
    their own preprocessing), so the cache does not amortise across
    replicates. It does amortise across everything else you might vary on the
    same genotypes: an alpha sweep, both knockoff rows, a second phenotype
    configuration, a different feature statistic. Given what the construction
    costs, that is the difference between rerunning the statistic in minutes
    and rerunning the whole thing in a day.
    """
    cache_dir = _cfg(cfg, "ko_cache_dir")
    path = None
    if cache_dir and run_id is not None:
        path = knockoff_cache_path(cache_dir, run_id, cfg)
        if path.exists():
            Xk = np.load(path)
            if Xk.shape == Xc.shape:
                _log(f"  knockpy: loaded cache {path.name}")
                return np.asarray(Xk, dtype=np.float64), {
                    "ko_backend": "knockpy", "ko_cached": True}
            _log(f"  knockpy: cache {path.name} has shape {Xk.shape}, "
                 f"expected {Xc.shape}; regenerating")

    Xk, info = sample_knockoffs(Xc, cfg)
    info["ko_cached"] = False
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:          # np.save would re-append ".npy"
            np.save(fh, Xk)
        os.replace(tmp, path)
        _log(f"  knockpy: cached to {path}")
    return Xk, info


# =====================================================================
# 2. The multinomial feature statistic
# =====================================================================

def _logreg(solver, C_val, max_iter, tol, seed):
    """An ell_1 logistic regression across sklearn versions.

    ``penalty="l1"`` is deprecated from 1.8 in favour of ``l1_ratio=1.0``, and
    ``multi_class`` was removed. Build the estimator with whichever the
    installed version accepts.
    """
    import inspect
    import warnings
    from sklearn.linear_model import LogisticRegression

    params = inspect.signature(LogisticRegression.__init__).parameters
    kw = dict(solver=solver, C=C_val, max_iter=max_iter, tol=tol,
              random_state=seed, fit_intercept=True)
    if "l1_ratio" in params:
        kw["l1_ratio"] = 1.0
    else:
        kw["penalty"] = "l1"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return LogisticRegression(**kw)
        except TypeError:
            kw.pop("l1_ratio", None)
            kw["penalty"] = "l1"
            return LogisticRegression(**kw)


def _fit_multinomial_l1(Xaug, y, C_val, max_iter, tol, seed):
    import warnings
    clf = _logreg("saga", C_val, max_iter, tol, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xaug, y)
    return np.atleast_2d(clf.coef_)          # (n_classes, 2p)


def _fit_ovr_l1(Xaug, y, C_val, max_iter, tol, seed):
    """One coefficient row per subtype, each from a binary ell_1 logistic fit.

    Cheaper than the multinomial MLE and still a valid knockoff statistic:
    swapping a column with its knockoff swaps the two coefficients in every
    per-class fit, so the aggregated Z flips as required.
    """
    import warnings
    y = np.asarray(y, dtype=int).ravel()
    rows = []
    for c in np.unique(y)[1:]:
        clf = _logreg("liblinear", C_val, max_iter, tol, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xaug, (y == c).astype(int))
        rows.append(clf.coef_.ravel())
    return np.vstack(rows)


_FITTERS = {"multinomial_l1": _fit_multinomial_l1, "ovr_l1": _fit_ovr_l1}

_AGG = {
    "l2": lambda B: np.linalg.norm(B, axis=0),
    "max": lambda B: np.abs(B).max(axis=0),
    "sum": lambda B: np.abs(B).sum(axis=0),
}


def _c_max(Xaug, y):
    """Smallest sklearn ``C`` at which every coefficient is still zero.

    At the intercept-only fit the score is ``Xaug' (Y - pihat)``, and the KKT
    condition for an all-zero solution is ``C * max |score| <= 1``. Anchoring
    the grid here makes it scale free, which matters because the HAPNEST
    columns are ell_2-normalised rather than standardised: knockpy's
    ``DEFAULT_REG_VALS`` are calibrated for standardised columns and would put
    the whole path in the all-zero region here.
    """
    y = np.asarray(y, dtype=int).ravel()
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(np.float64)
    R = Y - Y.mean(axis=0, keepdims=True)
    m = float(np.abs(Xaug.T @ R).max())
    return 1.0 / m if m > 0 else 1.0


def multinomial_lcd(Xc, Xk, y, cfg):
    """Lasso coefficient-difference statistic, aggregated across classes.

        W_j = || beta_j ||_2  -  || beta_j~ ||_2

    with ``beta_j`` the vector of that SNP's coefficients over the C-1
    non-reference classes. The ell_2 aggregation matches how the VD-AFS
    multinomial solver ranks candidates, which keeps the comparison honest;
    ``knockoff::stat.glmnet_coefdiff`` sums absolute values instead
    (``ko_agg="sum"``) and ``multinomial_baselines`` takes the max
    (``ko_agg="max"``). All three satisfy the flip-sign property.

    Careful with the order of operations. Aggregating first and differencing
    second flips sign under a column swap; differencing first and aggregating
    second does not, because ``max_c(-d_c) = -min_c(d_c)``. The docstring of
    ``multinomial_baselines.multinomial_lasso_knockoff_stats`` describes the
    second, its code does the first. The code is the valid one.

    Two knockpy conventions are followed. The ``2p`` columns are randomly
    permuted before fitting, which ``knockoff_stats.fit_lasso`` does so the
    solver cannot break ties by column position, and the antisymmetric
    combination is ``knockoff_stats.combine_Z_stats(Z, antisym="cd")`` rather
    than a hand-written subtraction.
    """
    from knockpy import knockoff_stats as ks
    from knockpy import utilities

    y = np.asarray(y, dtype=int).ravel()
    n, p = Xc.shape
    seed = int(cfg.get("ko_seed", DEFAULT_CFG["ko_seed"]))

    fstat = _cfg(cfg, "ko_fstat")
    if fstat not in _FITTERS:
        raise ValueError(f"ko_fstat must be one of {list(_FITTERS)}, "
                         f"got {fstat!r}")
    agg_name = _cfg(cfg, "ko_agg")
    if agg_name not in _AGG:
        raise ValueError(f"ko_agg must be one of {list(_AGG)}, "
                         f"got {agg_name!r}")
    fit = _FITTERS[fstat]
    max_iter, tol = _cfg(cfg, "ko_max_iter"), _cfg(cfg, "ko_tol")

    # knockpy's own trick: randomise the column order so the solver cannot
    # break ties by position, then undo the permutation on the coefficients
    np.random.seed(seed)
    inds, rev_inds = utilities.random_permutation_inds(2 * p)
    Xaug = np.hstack([np.asarray(Xc, dtype=np.float64),
                      np.asarray(Xk, dtype=np.float64)])[:, inds]

    # Regularisation path, anchored at the C where the first column enters
    n_lambda = int(_cfg(cfg, "ko_n_lambda"))
    span = float(_cfg(cfg, "ko_c_span"))
    C0 = _c_max(Xaug, y)
    grid = C0 * np.geomspace(1.02, span, max(n_lambda, 2))
    if n_lambda == 1:
        grid = np.array([C0 * np.sqrt(span)])
    n_folds = int(_cfg(cfg, "ko_cv"))

    if len(grid) == 1 or n_folds < 2:
        C_best = float(grid[len(grid) // 2])
        cv_scores = None
    else:
        import warnings
        from sklearn.metrics import log_loss
        from sklearn.model_selection import StratifiedKFold
        solver = "saga" if fstat == "multinomial_l1" else "liblinear"
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=seed).split(Xaug, y))
        labels = np.unique(y)

        # Xa is an explicit argument, not a closure capture. joblib memmaps
        # large ndarray *arguments* and pickles everything else, so capturing
        # Xaug here would hand every worker its own copy: at n = 10^4 and
        # 2p ~ 62.000 that is 5 GB per process, and ko_n_jobs = 16 asks the
        # scheduler for 80 GB it was never told about.
        def _one(Xa, ci, C_val, tr, te):
            clf = _logreg(solver, C_val, max_iter, tol, seed)
            if fstat == "ovr_l1":
                from sklearn.multiclass import OneVsRestClassifier
                clf = OneVsRestClassifier(clf)   # liblinear is binary only
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(Xa[tr], y[tr])
            return ci, log_loss(y[te], clf.predict_proba(Xa[te]),
                                labels=labels)

        jobs = [(ci, C_val, tr, te) for ci, C_val in enumerate(grid)
                for tr, te in folds]
        # saga and liblinear are single threaded, so the cores go into the CV.
        # The fold slice Xa[tr] is still a per-worker copy, ~2/3 of Xaug, so
        # the job needs roughly
        #     4 * Xaug   (parent: Xc, Xk, Xaug, the shared memmap)
        #   + n_jobs * (2/3 * Xaug + solver overhead)
        # Set JOBLIB_TEMP_FOLDER to node-local scratch, or joblib puts the
        # memmap in /dev/shm, which counts against the cgroup limit.
        n_jobs = int(_cfg(cfg, "ko_n_jobs"))
        if n_jobs == 1:
            out = [_one(Xaug, *j) for j in jobs]
        else:
            from joblib import Parallel, delayed
            out = Parallel(n_jobs=n_jobs, prefer="processes",
                           max_nbytes="64M", mmap_mode="r")(
                delayed(_one)(Xaug, *j) for j in jobs)
        cv_scores = [0.0] * len(grid)
        for ci, loss in out:
            cv_scores[ci] += loss / len(folds)
        C_best = float(grid[int(np.argmin(cv_scores))])

    t0 = time.perf_counter()

    def _Z_at(C_val):
        coef = fit(Xaug, y, C_val, max_iter, tol, seed)
        return _AGG[agg_name](coef)[rev_inds]     # undo the permutation

    Z = _Z_at(C_best)

    # The CV can land on the intercept-only model, in which case Z is
    # identically zero, W is identically zero, and the row reports FDP 0 and
    # TPP 0 without saying anything. Step up the path to the first C that
    # admits a nonzero coefficient. This stays a valid feature statistic: the
    # rule depends on the augmented matrix only through quantities that are
    # invariant under swapping a column with its knockoff.
    n_bumps = 0
    larger = [c for c in grid if c > C_best]
    while not np.any(Z > 0) and n_bumps < len(larger):
        C_best = float(larger[n_bumps])
        Z = _Z_at(C_best)
        n_bumps += 1

    W = ks.combine_Z_stats(Z, antisym="cd")
    info = {
        "ko_fstat": fstat, "ko_agg": agg_name, "ko_C": C_best,
        "ko_C_max": float(C0), "ko_c_span": span, "ko_bumps": n_bumps,
        "ko_n_lambda": n_lambda, "ko_cv_folds": n_folds,
        "ko_nonzero": int(np.sum(Z > 0)),
        "ko_fit_seconds": round(time.perf_counter() - t0, 2),
    }
    if cv_scores is not None:
        info["ko_cv_scores"] = json.dumps([round(v, 5) for v in cv_scores])
    return W, info


def knockoff_threshold_plus(W, q):
    """Knockoff+ threshold, i.e. knockpy's ``data_dependent_threshhold`` with
    offset 1. Returns ``inf`` when nothing can be rejected.
    """
    from knockpy import knockoff_stats as ks
    t = float(ks.data_dependent_threshhold(np.asarray(W, dtype=float),
                                           fdr=q, offset=1))
    return np.inf if not np.isfinite(t) else t


# =====================================================================
# 3. Methods for benchmark_worker.DISPATCH
# =====================================================================

def _select(W, alpha):
    tau = knockoff_threshold_plus(W, alpha)
    if not np.isfinite(tau):
        return np.array([], dtype=int), None
    return np.where(W >= tau)[0].astype(int), round(tau, 8)


def m_ko_mx_multinom(Xc, X_raw, y, cfg, mmap_fd):
    """Model-X knockoffs on all p columns, multinomial LCD statistic."""
    Xc = np.asarray(Xc, dtype=np.float64)
    Xk, ko_info = build_or_load_knockoffs(Xc, cfg, run_id=cfg.get("_run_id"))
    W, w_info = multinomial_lcd(Xc, Xk, y, cfg)
    sel, tau = _select(W, cfg["alpha"])
    return sel, {**ko_info, **w_info, "ko_type": "model-X", "ko_tau": tau}


KO_DISPATCH = {
    "ko_mx_multinom": m_ko_mx_multinom,
}


# =====================================================================
# 4. CLI
# =====================================================================

def _cli_generate(argv):
    """Precompute and cache the knockoff matrix for one replicate."""
    data_dir, run_id, cache_dir = argv[0], int(argv[1]), argv[2]
    cfg = json.loads(argv[3]) if len(argv) > 3 else {}
    cfg.setdefault("ko_cache_dir", cache_dir)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmark_worker import load_genotypes

    loaded = load_genotypes(data_dir, run_id)
    if loaded is None:
        raise SystemExit(f"genotypes for run {run_id} not found in {data_dir}")
    Xc, X_raw, mmap_fd, mode, holders = loaded
    Xc = np.asarray(Xc, dtype=np.float64)
    _log(f"run={run_id} n={Xc.shape[0]} p={Xc.shape[1]} load={mode}")

    Xk, info = build_or_load_knockoffs(Xc, cfg, run_id=run_id)
    path = knockoff_cache_path(cache_dir, run_id, cfg)
    print(json.dumps({"run": run_id, "path": str(path), **info}, indent=1))
    del holders


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
