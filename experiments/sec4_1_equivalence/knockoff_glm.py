#!/usr/bin/env python
"""
Model-X knockoff competitor of Sec. 4.1: second-order Gaussian knockoffs and a
lasso coefficient-difference statistic in the generating family.

The construction depends on X only and is knockpy's
(estimate_covariance with Ledoit-Wolf shrinkage, equicorrelated S-matrix,
GaussianSampler). The statistic W_j = |beta_j| - |beta_{j+p}| comes from an
ell_1-penalised fit on [X, X_tilde] in the family of the response: Poisson
(skglm), Bernoulli (sklearn), multinomial (sklearn) or Cox (scikit-survival).
The penalty is chosen by cross-validation on a path anchored at the value
where the first column enters, because the columns are unit-norm rather than
standardised. The knockoff+ threshold gives the selection.

    glm_knockoff(X, y, family, alpha=0.1, seed=0, cfg=None)
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
from __future__ import annotations

import time
import numpy as np

__all__ = ["DEFAULT_CFG", "sample_knockoffs", "knockoff_threshold",
           "poisson_lcd", "logistic_lcd", "multinomial_lcd", "cox_lcd",
           "glm_knockoff", "FAMILIES"]


# ================================================================
# Configuration
# ================================================================
DEFAULT_CFG = dict(
    # construction
    ko_shrinkage="ledoitwolf",
    ko_cov_tol=1e-2,
    ko_method="equicorrelated",   # None -> knockpy's default (mvr)
    ko_how_approx="blockdiag",
    ko_max_block=1000,
    ko_num_processes=1,
    # statistic
    ko_n_lambda=25,
    ko_alpha_start=0.98,          # path starts just below alpha_max
    ko_alpha_span=0.005,          # path runs to ko_alpha_span * alpha_max
    ko_cv=5,
    ko_max_iter=50,
    ko_tol=1e-6,
    ko_eta_clip=30.0,             # guard exp() on held-out folds
    # statistic, Bernoulli only. sklearn parametrises by C = 1/penalty, so
    # its grid runs the other way: from ko_c_start * C_max upwards.
    ko_fstat="liblinear",         # or "saga" (unpenalised intercept, slower)
    ko_c_start=1.02,
    ko_c_span=200.0,
    ko_lr_max_iter=200,
    ko_lr_tol=1e-4,
    # statistic, multinomial only
    ko_mfstat="ovr_l1",           # or "multinomial_l1" (true MLE, ~8x slower)
    ko_agg="l2",                  # aggregation across classes: l2, max, sum
    # statistic, Cox only. Coxnet derives its own penalty path from the data,
    # so there is nothing to anchor by hand here.
    ko_cox_n_alphas=30,
    ko_cox_alpha_min_ratio=0.05,
    ko_cox_max_iter=10000,
    ko_cox_tol=1e-6,
    ko_coef_tol=1e-8,
    # filter
    ko_offset=1,                  # 1 = knockoff+ (FDR control)
)

EPS = 1e-12
_VERBOSE = False


def _log(msg):
    if _VERBOSE:
        print(msg, flush=True)


def _cfg(cfg, key):
    if cfg is None:
        return DEFAULT_CFG[key]
    return cfg.get(key, DEFAULT_CFG[key])


# ================================================================
# 1. Construction  (entirely knockpy)
# ================================================================
def sample_knockoffs(Xc, cfg=None, seed=0, verbose=False):
    """Second-order model-X Gaussian knockoffs, entirely via knockpy.

    Parameters
    ----------
    Xc : (n, p) ndarray
        Column-centered, ell_2-normalised design, i.e. exactly what
        ``center_unitL2`` in the Figure 4 scripts returns.
    cfg : dict or None
        Reads ``ko_shrinkage``, ``ko_cov_tol``, ``ko_method``,
        ``ko_how_approx``, ``ko_max_block``, ``ko_num_processes``.
    seed : int
        ``GaussianSampler`` draws from numpy's global stream, so the stream is
        seeded here and restored afterwards. Without the restore this function
        would silently reseed its caller.

    Returns
    -------
    Xk : (n, p) ndarray on the same column scale as ``Xc``
    info : dict

    Notes
    -----
    knockpy works in the unit-variance scale, so the design goes in as
    ``Z = sqrt(n) Xc`` and the result is divided back. No per-column
    renormalisation afterwards: that would break pairwise exchangeability.
    """
    from knockpy import smatrix, utilities
    from knockpy.knockoffs import GaussianSampler

    Xc = np.asarray(Xc, dtype=np.float64)
    n, p = Xc.shape
    t0 = time.perf_counter()
    Z = np.sqrt(n) * Xc

    shrinkage = _cfg(cfg, "ko_shrinkage")
    Sigma, invSigma = utilities.estimate_covariance(
        Z, _cfg(cfg, "ko_cov_tol"), shrinkage)
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

    state = np.random.get_state()
    try:
        np.random.seed(int(seed) % (2**32 - 1))
        sampler = GaussianSampler(Z, mu=np.zeros(p), Sigma=Sigma,
                                  invSigma=invSigma, S=S)
        Xk = sampler.sample_knockoffs() / np.sqrt(n)
    finally:
        np.random.set_state(state)

    s_diag = np.diag(S)
    info = {
        "ko_backend": "knockpy",
        "ko_shrinkage": shrinkage,
        "ko_method": "mvr (knockpy default)" if method is None else str(method),
        "ko_s_min": float(s_diag.min()), "ko_s_max": float(s_diag.max()),
        "ko_s_mean": float(s_diag.mean()),
        "ko_n": int(n), "ko_p": int(p),
        "ko_cov_seconds": round(t_cov, 3),
        "ko_smatrix_seconds": round(t_s, 3),
        "ko_gen_seconds": round(time.perf_counter() - t0, 3),
    }
    if verbose or _VERBOSE:
        _log(f"  knockpy: n={n} p={p} method={info['ko_method']} "
             f"s in [{info['ko_s_min']:.4f}, {info['ko_s_max']:.4f}] "
             f"({info['ko_gen_seconds']:.1f}s)")
    return Xk, info


# ================================================================
# 2. Poisson lasso coefficient difference
# ================================================================
def _alpha_max(Xaug, y):
    """Smallest penalty at which every coefficient is still zero.

    At the intercept-only fit ``b0 = log ybar`` the gradient of the datafit is
    ``Xaug' (ybar - y) / n``, and the KKT condition for an all-zero solution
    is ``alpha >= max_j |.|``. Anchoring the grid here makes it scale free,
    which is what the ell_2-normalised columns require.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = Xaug.shape[0]
    m = float(np.abs(Xaug.T @ (y - y.mean())).max()) / n
    return m if m > 0 else 1.0


def _estimator(alpha, cfg):
    from skglm import GeneralizedLinearEstimator
    from skglm.datafits import Poisson
    from skglm.penalties import L1
    from skglm.solvers import ProxNewton
    return GeneralizedLinearEstimator(
        datafit=Poisson(),
        penalty=L1(alpha=float(alpha)),
        solver=ProxNewton(fit_intercept=True,
                          tol=float(_cfg(cfg, "ko_tol")),
                          max_iter=int(_cfg(cfg, "ko_max_iter")),
                          warm_start=True),
    )


def _fit_path(X, y, grid, cfg):
    """Warm-started Poisson lasso down ``grid``. Returns list of (coef, b0)."""
    est = _estimator(grid[0], cfg)
    out = []
    for a in grid:
        est.penalty.alpha = float(a)
        est.fit(X, y)
        out.append((np.asarray(est.coef_, dtype=np.float64).ravel().copy(),
                    float(np.ravel(est.intercept_)[0])))
        est.solver.warm_start = True
    return out


def _poisson_nll(X, y, coef, b0, clip):
    """Mean negative Poisson log-likelihood, constants dropped."""
    eta = np.clip(X @ coef + b0, -clip, clip)
    return float(np.mean(np.exp(eta) - y * eta))


def poisson_lcd(Xc, Xk, y, cfg=None, seed=0):
    """Lasso coefficient-difference statistic for a Poisson response.

        W_j = |beta_j| - |beta_{j+p}|

    at the cross-validated penalty. Returns ``(W, info)``.

    The ``2p`` columns are permuted before the fit and the coefficients
    permuted back afterwards, following ``knockpy.knockoff_stats.fit_lasso``.
    The antisymmetric combination is knockpy's ``combine_Z_stats`` with
    ``antisym="cd"`` rather than a hand-written subtraction.
    """
    from knockpy import knockoff_stats as ks
    from knockpy import utilities

    Xc = np.asarray(Xc, dtype=np.float64)
    Xk = np.asarray(Xk, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = Xc.shape
    t0 = time.perf_counter()

    # knockpy's own trick: randomise the column order so the solver cannot
    # break ties by position, then undo the permutation on the coefficients.
    rng_state = np.random.get_state()
    try:
        np.random.seed(int(seed) % (2**32 - 1))
        inds, rev_inds = utilities.random_permutation_inds(2 * p)
    finally:
        np.random.set_state(rng_state)
    Xaug = np.hstack([Xc, Xk])[:, inds]

    n_lambda = max(int(_cfg(cfg, "ko_n_lambda")), 1)
    span = float(_cfg(cfg, "ko_alpha_span"))
    clip = float(_cfg(cfg, "ko_eta_clip"))
    a_max = _alpha_max(Xaug, y)
    start = float(_cfg(cfg, "ko_alpha_start"))
    grid = a_max * np.geomspace(start, span, n_lambda)

    n_folds = int(_cfg(cfg, "ko_cv"))
    if n_lambda == 1 or n_folds < 2:
        a_idx, cv_scores = n_lambda // 2, None
    else:
        from sklearn.model_selection import KFold
        folds = list(KFold(n_splits=n_folds, shuffle=True,
                           random_state=int(seed) % (2**31)).split(Xaug))
        scores = np.zeros((n_folds, n_lambda))
        for f, (tr, te) in enumerate(folds):
            path = _fit_path(Xaug[tr], y[tr], grid, cfg)
            for i, (coef, b0) in enumerate(path):
                scores[f, i] = _poisson_nll(Xaug[te], y[te], coef, b0, clip)
        cv_scores = scores.mean(axis=0)
        a_idx = int(np.argmin(cv_scores))

    # Full-data path, so the selected fit is the one the CV pointed at.
    path = _fit_path(Xaug, y, grid, cfg)
    coefs = [c for c, _ in path]

    # If the CV lands on the all-zero fit, walk down to the first alpha with a
    # nonzero coefficient. The rule depends only on swap-invariant quantities,
    # so the statistic stays valid.
    bumps = 0
    while a_idx < n_lambda - 1 and not np.any(coefs[a_idx] != 0):
        a_idx += 1
        bumps += 1

    beta = coefs[a_idx][rev_inds]          # undo the permutation
    Z = np.abs(beta)
    W = ks.combine_Z_stats(Z, antisym="cd")

    info = {
        "ko_alpha": float(grid[a_idx]),
        "ko_alpha_max": float(a_max),
        "ko_alpha_idx": int(a_idx),
        "ko_n_lambda": int(n_lambda),
        "ko_bumps": int(bumps),
        "ko_nonzero": int(np.sum(beta != 0)),
        "ko_cv": int(n_folds),
        "ko_stat_seconds": round(time.perf_counter() - t0, 3),
    }
    if cv_scores is not None:
        info["ko_cv_nll"] = float(cv_scores[min(a_idx, n_lambda - 1)])
    return np.asarray(W, dtype=np.float64).ravel(), info


# ================================================================
# 2b. Bernoulli lasso coefficient difference
# ================================================================
def _logreg(solver, C_val, max_iter, tol, seed, enet=False):
    """An ell_1 logistic regression, with a guard against a silent ridge fit.

    ``knockoff_multinomial._logreg`` builds this by testing whether
    ``l1_ratio`` is in the signature and, if so, passing ``l1_ratio=1.0``
    INSTEAD of ``penalty="l1"``. That test is always true -- ``l1_ratio`` has
    been a parameter since sklearn 0.21 -- but ``l1_ratio`` is only read when
    ``penalty="elasticnet"``, so the estimator falls back to the default
    ``penalty="l2"`` and fits a RIDGE. sklearn even warns about it, and the
    warning is swallowed by the surrounding ``simplefilter("ignore")``. The
    coefficients are then all nonzero and the statistic, while still valid
    (the flip-sign property does not care which penalty was used), is not the
    lasso coefficient difference the method calls for and is far weaker.

    So: ask for ``penalty="l1"``, which every current sklearn accepts, and
    keep the elasticnet spelling as the fallback for a future release that
    removes it. ``poisson_lcd`` is unaffected -- it does not go through
    sklearn at all. Test B12 checks that the fit really is sparse.
    """
    import warnings
    from sklearn.linear_model import LogisticRegression

    kw = dict(solver=solver, C=float(C_val), max_iter=int(max_iter),
              tol=float(tol), random_state=int(seed), fit_intercept=True)
    kw.update(dict(penalty="elasticnet", l1_ratio=1.0) if enet
              else dict(penalty="l1"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(**kw)


def _logreg_fit(Xaug, y, C_val, cfg, seed):
    """Fit, falling back to the elasticnet spelling if ``l1`` is refused."""
    import warnings
    solver = str(_cfg(cfg, "ko_fstat"))
    mi, tol = _cfg(cfg, "ko_lr_max_iter"), _cfg(cfg, "ko_lr_tol")
    for enet in (False, True):
        clf = _logreg(solver if not enet else "saga", C_val, mi, tol, seed,
                      enet=enet)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(Xaug, np.asarray(y, dtype=int).ravel())
            return clf
        except (ValueError, TypeError):
            if enet:
                raise
    raise RuntimeError("unreachable")


def _binary_labels(y):
    """Map a two-valued response to 0/1, whatever the coding.

    The Figure 4 logistic sweep hands its arms a CENTERED binary response --
    ``sample_y_snr`` subtracts the prevalence, so the two values are
    ``-pbar`` and ``1-pbar``, not 0 and 1. Rather than assume either coding,
    take the two distinct values and call the larger one 1. Nothing else in
    the statistic changes: the anchor uses ``y - ybar``, which is identical
    under both codings.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    vals = np.unique(y)
    if vals.size != 2:
        raise ValueError(
            f"a Bernoulli response must take exactly two values, got "
            f"{vals.size} ({vals[:5]}). A replicate with one class only is "
            f"degenerate, not something this statistic can rank.")
    return (y == vals[1]).astype(np.int64)


def _c_max_binom(Xaug, y):
    """Largest sklearn ``C`` at which every coefficient is still zero.

    sklearn minimises ``C * sum_i loss_i + ||beta||_1``, so the KKT condition
    at the intercept-only fit ``b0 = logit(ybar)`` reads
    ``C * max_j |x_j' (y - ybar)| <= 1``. Same anchoring idea as
    ``_alpha_max`` for Poisson, mirrored because C is an inverse penalty:
    below this value nothing enters, above it the first column does.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    m = float(np.abs(Xaug.T @ (y - y.mean())).max())
    return 1.0 / m if m > 0 else 1.0


def _binom_fit(Xaug, y, C_val, cfg, seed):
    clf = _logreg_fit(Xaug, y, C_val, cfg, seed)
    return np.asarray(clf.coef_, dtype=np.float64).ravel()


def logistic_lcd(Xc, Xk, y, cfg=None, seed=0):
    """Lasso coefficient-difference statistic for a Bernoulli response.

        W_j = |beta_j(C)| - |beta_{j+p}(C)|

    at the cross-validated penalty. Returns ``(W, info)``.

    Structurally identical to ``poisson_lcd``: same column permutation, same
    anchored grid, same bump-up rule when the cross-validation lands on the
    intercept-only model, same antisymmetric combination via knockpy. Only
    the fit and the parametrisation of the penalty differ, and the grid runs
    the other way because sklearn's ``C`` is an inverse penalty.
    """
    from knockpy import knockoff_stats as ks
    from knockpy import utilities

    Xc = np.asarray(Xc, dtype=np.float64)
    Xk = np.asarray(Xk, dtype=np.float64)
    yi = _binary_labels(y)
    n, p = Xc.shape
    t0 = time.perf_counter()

    rng_state = np.random.get_state()
    try:
        np.random.seed(int(seed) % (2**32 - 1))
        inds, rev_inds = utilities.random_permutation_inds(2 * p)
    finally:
        np.random.set_state(rng_state)
    Xaug = np.hstack([Xc, Xk])[:, inds]

    n_lambda = max(int(_cfg(cfg, "ko_n_lambda")), 1)
    start = float(_cfg(cfg, "ko_c_start"))
    span = float(_cfg(cfg, "ko_c_span"))
    C0 = _c_max_binom(Xaug, yi)
    grid = C0 * np.geomspace(start, span, n_lambda)

    n_folds = int(_cfg(cfg, "ko_cv"))
    if n_lambda == 1 or n_folds < 2 or min(np.bincount(yi, minlength=2)) < n_folds:
        c_idx, cv_scores = n_lambda // 2, None
    else:
        from sklearn.metrics import log_loss
        from sklearn.model_selection import StratifiedKFold
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=int(seed) % (2**31)
                                     ).split(Xaug, yi))
        scores = np.zeros((n_folds, n_lambda))
        for f, (tr, te) in enumerate(folds):
            for i, C_val in enumerate(grid):
                clf = _logreg_fit(Xaug[tr], yi[tr], C_val, cfg, int(seed))
                scores[f, i] = log_loss(yi[te], clf.predict_proba(Xaug[te]),
                                        labels=[0, 1])
        cv_scores = scores.mean(axis=0)
        c_idx = int(np.argmin(cv_scores))

    beta = _binom_fit(Xaug, yi, grid[c_idx], cfg, seed)
    bumps = 0
    while c_idx < n_lambda - 1 and not np.any(beta != 0):
        c_idx += 1
        bumps += 1
        beta = _binom_fit(Xaug, yi, grid[c_idx], cfg, seed)

    Z = np.abs(beta[rev_inds])
    W = ks.combine_Z_stats(Z, antisym="cd")

    info = {
        "ko_alpha": float(grid[c_idx]),          # here: the C, not a penalty
        "ko_alpha_max": float(C0),
        "ko_alpha_idx": int(c_idx),
        "ko_n_lambda": int(n_lambda),
        "ko_bumps": int(bumps),
        "ko_nonzero": int(np.sum(beta != 0)),
        "ko_cv": int(n_folds),
        "ko_stat_seconds": round(time.perf_counter() - t0, 3),
    }
    if cv_scores is not None:
        info["ko_cv_nll"] = float(cv_scores[min(c_idx, n_lambda - 1)])
    return np.asarray(W, dtype=np.float64).ravel(), info


# ================================================================
# 2c. Multinomial lasso coefficient difference
# ================================================================
_AGG = {
    "l2": lambda B: np.linalg.norm(B, axis=0),
    "max": lambda B: np.abs(B).max(axis=0),
    "sum": lambda B: np.abs(B).sum(axis=0),
}


def _c_max_multinom(Xaug, y):
    """Largest sklearn ``C`` at which every coefficient is still zero.

    At the intercept-only fit the score is ``Xaug' (Y - pibar)`` with ``Y``
    the one-hot response, so the KKT condition reads
    ``C * max |score| <= 1``. Same rule as
    ``knockoff_multinomial._c_max``.
    """
    y = np.asarray(y, dtype=int).ravel()
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(np.float64)
    R = Y - Y.mean(axis=0, keepdims=True)
    m = float(np.abs(Xaug.T @ R).max())
    return 1.0 / m if m > 0 else 1.0


def _multinom_fit(Xaug, y, C_val, cfg, seed):
    """Coefficient matrix (C-1 or C rows, 2p columns) from an ell_1 fit."""
    import warnings
    y = np.asarray(y, dtype=int).ravel()
    mi, tol = _cfg(cfg, "ko_lr_max_iter"), _cfg(cfg, "ko_lr_tol")
    if str(_cfg(cfg, "ko_mfstat")) == "multinomial_l1":
        clf = _logreg("saga", C_val, mi, tol, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xaug, y)
        return np.atleast_2d(np.asarray(clf.coef_, dtype=np.float64)), clf
    # One-vs-rest against the reference class. Cheaper than the multinomial
    # MLE and still a valid knockoff statistic: swapping a column with its
    # knockoff swaps the two coefficients in every per-class fit, so the
    # aggregated Z flips as required.
    rows = []
    for c in np.unique(y)[1:]:
        clf = _logreg("liblinear", C_val, mi, tol, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xaug, (y == c).astype(int))
        rows.append(np.asarray(clf.coef_, dtype=np.float64).ravel())
    return np.vstack(rows), None


def multinomial_lcd(Xc, Xk, y, cfg=None, seed=0):
    """Lasso coefficient-difference statistic, aggregated across classes.

        W_j = || beta_j ||_2  -  || beta_j~ ||_2

    with ``beta_j`` the vector of that feature's coefficients over the
    non-reference classes. The ell_2 aggregation matches how the VD-AFS
    multinomial solver ranks candidates, which keeps the comparison honest;
    ``knockoff::stat.glmnet_coefdiff`` sums absolute values instead
    (``ko_agg="sum"``) and ``multinomial_baselines`` takes the max.

    Careful with the order of operations. Aggregating first and differencing
    second flips sign under a column swap; differencing first and aggregating
    second does not, because ``max_c(-d_c) = -min_c(d_c)``. This does the
    first, like ``knockoff_multinomial.multinomial_lcd``.
    """
    from knockpy import knockoff_stats as ks
    from knockpy import utilities

    Xc = np.asarray(Xc, dtype=np.float64)
    Xk = np.asarray(Xk, dtype=np.float64)
    yi = np.asarray(y, dtype=int).ravel()
    n, p = Xc.shape
    t0 = time.perf_counter()

    agg = str(_cfg(cfg, "ko_agg"))
    if agg not in _AGG:
        raise ValueError(f"ko_agg must be one of {list(_AGG)}, got {agg!r}")

    rng_state = np.random.get_state()
    try:
        np.random.seed(int(seed) % (2**32 - 1))
        inds, rev_inds = utilities.random_permutation_inds(2 * p)
    finally:
        np.random.set_state(rng_state)
    Xaug = np.hstack([Xc, Xk])[:, inds]

    n_lambda = max(int(_cfg(cfg, "ko_n_lambda")), 1)
    grid = _c_max_multinom(Xaug, yi) * np.geomspace(
        float(_cfg(cfg, "ko_c_start")), float(_cfg(cfg, "ko_c_span")), n_lambda)
    C0 = float(grid[0])

    n_folds = int(_cfg(cfg, "ko_cv"))
    counts = np.bincount(yi)
    if n_lambda == 1 or n_folds < 2 or counts[counts > 0].min() < n_folds:
        c_idx, cv_scores = n_lambda // 2, None
    else:
        from sklearn.metrics import log_loss
        from sklearn.model_selection import StratifiedKFold
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=int(seed) % (2**31)
                                     ).split(Xaug, yi))
        labels = np.unique(yi)
        scores = np.zeros((n_folds, n_lambda))
        for f, (tr, te) in enumerate(folds):
            for i, C_val in enumerate(grid):
                B, clf = _multinom_fit(Xaug[tr], yi[tr], C_val, cfg, int(seed))
                if clf is not None:
                    proba = clf.predict_proba(Xaug[te])
                else:
                    # one-vs-rest: reconstruct class probabilities from the
                    # per-class scores, normalised across classes
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        z = np.column_stack(
                            [np.zeros(len(te))] +
                            [Xaug[te] @ B[k] for k in range(B.shape[0])])
                    z -= z.max(axis=1, keepdims=True)
                    e = np.exp(z)
                    proba = e / e.sum(axis=1, keepdims=True)
                scores[f, i] = log_loss(yi[te], proba, labels=labels)
        cv_scores = scores.mean(axis=0)
        c_idx = int(np.argmin(cv_scores))

    B, _ = _multinom_fit(Xaug, yi, grid[c_idx], cfg, seed)
    bumps = 0
    while c_idx < n_lambda - 1 and not np.any(B != 0):
        c_idx += 1
        bumps += 1
        B, _ = _multinom_fit(Xaug, yi, grid[c_idx], cfg, seed)

    Z = _AGG[agg](B)[rev_inds]
    W = ks.combine_Z_stats(Z, antisym="cd")

    info = {
        "ko_alpha": float(grid[c_idx]), "ko_alpha_max": C0,
        "ko_alpha_idx": int(c_idx), "ko_n_lambda": int(n_lambda),
        "ko_bumps": int(bumps), "ko_nonzero": int(np.sum(Z > 0)),
        "ko_cv": int(n_folds), "ko_agg": agg,
        "ko_stat_seconds": round(time.perf_counter() - t0, 3),
    }
    if cv_scores is not None:
        info["ko_cv_nll"] = float(cv_scores[min(c_idx, n_lambda - 1)])
    return np.asarray(W, dtype=np.float64).ravel(), info


# ================================================================
# 2d. Cox lasso coefficient difference -- CoxKnockoff
#
# Li, Yu and Zhao (2023), "Controlled feature selection for the Cox model
# using knockoffs". The construction is the second-order Gaussian model-X one
# of section 1 with the equicorrelated s of that paper, and the statistic is
# the coefficient difference of a penalised partial-likelihood fit on
# [X X_tilde], i.e. the Cox case of knockoff::stat.glmnet_coefdiff. Both
# halves therefore already exist: this function is the second one, and it is
# the same construction ``icassp_mehler/cox_knockoff.py`` runs on KIRC.
#
# The fit is scikit-survival's Coxnet with l1_ratio = 1, which is glmnet's
# Cox coordinate descent. Coxnet derives its own penalty path from the data,
# so unlike the Poisson and logistic statistics there is nothing to anchor by
# hand -- the ell_2-normalised columns cannot push the grid into the all-zero
# region here. The path is computed once on the full data and the SAME grid is
# reused inside the folds, so the cross-validated choice is comparable across
# folds; the rule is "min" on the held-out concordance, not "1se", because
# knockoffs threshold afterwards and an extra standard error of shrinkage in
# the statistic only costs power.
# ================================================================
def _surv(y):
    """Split the (n, 2) response into times and event indicators."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("a Cox response must be an (n, 2) array of "
                         "[time, event], got shape " + str(y.shape))
    return y[:, 0], y[:, 1].astype(bool)


def _coxnet_path(X, time_, event, alphas, cfg):
    """Coefficient path on a fixed (or self-chosen) penalty grid."""
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv
    ys = Surv.from_arrays(event=np.asarray(event, dtype=bool),
                          time=np.asarray(time_, dtype=np.float64))
    kw = dict(l1_ratio=1.0, normalize=False, fit_baseline_model=False,
              max_iter=int(_cfg(cfg, "ko_cox_max_iter")),
              tol=float(_cfg(cfg, "ko_cox_tol")))
    if alphas is None:
        kw.update(n_alphas=int(_cfg(cfg, "ko_cox_n_alphas")),
                  alpha_min_ratio=float(_cfg(cfg, "ko_cox_alpha_min_ratio")))
    else:
        kw["alphas"] = np.asarray(alphas, dtype=np.float64)
    grid = np.asarray(alphas, dtype=np.float64) if alphas is not None else None
    while True:
        try:
            m = CoxnetSurvivalAnalysis(**kw)
            m.fit(np.asarray(X, dtype=np.float64), ys)
            coef = np.asarray(m.coef_, dtype=np.float64)
            return np.asarray(m.alphas_, dtype=np.float64), (
                coef[:, None] if coef.ndim == 1 else coef)
        except Exception:                                  # noqa: BLE001
            # Coxnet raises at the weak-penalty end when a fold is degenerate.
            # Drop the smallest penalties and retry; the filter only needs the
            # strong-penalty end. Same fallback as _coxnet_coef_path in
            # sec4_4_kirc/kirc_selection.py.
            if grid is None or grid.size < 4:
                raise
            grid = grid[: max(2, grid.size // 2)]
            kw["alphas"] = grid


def cox_lcd(Xc, Xk, y, cfg=None, seed=0):
    """CoxKnockoff's lasso coefficient difference.

        W_j = |beta_j(lambda)| - |beta_{j+p}(lambda)|

    ``y`` is an ``(n, 2)`` array of ``[time, event]``. Returns ``(W, info)``.
    """
    from knockpy import knockoff_stats as ks
    from knockpy import utilities
    from sksurv.metrics import concordance_index_censored

    Xc = np.asarray(Xc, dtype=np.float64)
    Xk = np.asarray(Xk, dtype=np.float64)
    time_, event = _surv(y)
    n, p = Xc.shape
    t0 = time.perf_counter()

    rng_state = np.random.get_state()
    try:
        np.random.seed(int(seed) % (2**32 - 1))
        inds, rev_inds = utilities.random_permutation_inds(2 * p)
    finally:
        np.random.set_state(rng_state)
    Xaug = np.hstack([Xc, Xk])[:, inds]

    grid, path = _coxnet_path(Xaug, time_, event, None, cfg)
    n_lambda = int(path.shape[1])
    coef_tol = float(_cfg(cfg, "ko_coef_tol"))

    n_folds = int(_cfg(cfg, "ko_cv"))
    if n_lambda == 1 or n_folds < 2 or int(event.sum()) < 2 * n_folds:
        a_idx, cv_scores = n_lambda // 2, None
    else:
        from sklearn.model_selection import KFold
        folds = list(KFold(n_splits=n_folds, shuffle=True,
                           random_state=int(seed) % (2**31)).split(Xaug))
        scores = np.full((n_folds, n_lambda), np.nan)
        for f, (tr, te) in enumerate(folds):
            if event[tr].sum() < 2 or event[te].sum() < 2:
                continue
            g_tr, path_tr = _coxnet_path(Xaug[tr], time_[tr], event[tr],
                                         grid, cfg)
            for i in range(min(path_tr.shape[1], n_lambda)):
                risk = Xaug[te] @ path_tr[:, i]
                if not np.any(np.abs(path_tr[:, i]) > coef_tol):
                    scores[f, i] = 0.5           # intercept-only, no ranking
                    continue
                scores[f, i] = concordance_index_censored(
                    event[te], time_[te], risk)[0]
        with np.errstate(invalid="ignore"):
            cv_scores = np.nanmean(scores, axis=0)
        a_idx = (int(np.nanargmax(cv_scores))
                 if np.any(np.isfinite(cv_scores)) else n_lambda // 2)

    coef = path[:, a_idx]
    bumps = 0
    # The CV can land on a penalty at which everything is zero. Then W is
    # identically zero and the row reports FDP 0 / TPP 0 without saying
    # anything. Walk towards weaker penalties until something enters. The rule
    # reads the augmented matrix only through swap-invariant quantities, so
    # the statistic stays valid.
    while a_idx < n_lambda - 1 and not np.any(np.abs(coef) > coef_tol):
        a_idx += 1
        bumps += 1
        coef = path[:, a_idx]

    Z = np.abs(coef)[rev_inds]
    W = ks.combine_Z_stats(Z, antisym="cd")

    info = {
        "ko_alpha": float(grid[a_idx]), "ko_alpha_max": float(grid[0]),
        "ko_alpha_idx": int(a_idx), "ko_n_lambda": int(n_lambda),
        "ko_bumps": int(bumps),
        "ko_nonzero": int(np.sum(Z > coef_tol)),
        "ko_cv": int(n_folds),
        "ko_stat_seconds": round(time.perf_counter() - t0, 3),
    }
    if cv_scores is not None and np.isfinite(cv_scores[a_idx]):
        info["ko_cv_cindex"] = float(cv_scores[a_idx])
    return np.asarray(W, dtype=np.float64).ravel(), info


# ================================================================
# 3. Filter
# ================================================================
def knockoff_threshold(W, q, offset=1):
    """Knockoff (offset 0) / knockoff+ (offset 1) data-dependent threshold.

    Thin wrapper around ``knockpy.knockoff_stats.data_dependent_threshhold``,
    kept so the caller does not have to import knockpy and so the test suite
    has one place to check against the brute-force definition.
    """
    from knockpy import knockoff_stats as ks
    tau = ks.data_dependent_threshhold(
        np.asarray(W, dtype=np.float64).reshape(-1, 1),
        fdr=float(q), offset=int(offset))
    return float(np.ravel(np.asarray(tau))[0])


def _select(W, alpha, offset):
    tau = knockoff_threshold(W, alpha, offset)
    if not np.isfinite(tau):
        return np.array([], dtype=int), float("inf")
    return np.flatnonzero(W >= tau).astype(int), float(tau)


# ================================================================
# 4. The selector
# ================================================================
FAMILIES = {"poisson": poisson_lcd, "bernoulli": logistic_lcd,
            "multinomial": multinomial_lcd, "cox": cox_lcd}


def glm_knockoff(X, y, family, alpha=0.1, seed=0, cfg=None, **kw):
    """Model-X knockoffs on all p columns with the family's LCD statistic.

    Parameters
    ----------
    X : (n, p) ndarray, column-centered and ell_2-normalised
    y : the response, shaped by family -- counts for "poisson", two-valued
        for "bernoulli" (any coding), integer class labels for "multinomial",
        an (n, 2) array of [time, event] for "cox"
    family : "poisson", "bernoulli", "multinomial" or "cox"
    alpha : float, target level
    seed : int
    cfg : dict or None, overrides on top of ``DEFAULT_CFG``
    **kw : further ``ko_*`` overrides, merged after ``cfg``

    Returns
    -------
    dict with ``selected`` (knockoff+ selection), the threshold ``tau``, the
    statistic ``W`` and the fit details of the family's statistic.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, "
                         f"got {family!r}")
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    cfg.update(kw)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if family != "cox":
        y = y.ravel()
    t0 = time.perf_counter()

    Xk, info_k = sample_knockoffs(X, cfg, seed=seed)
    W, info_s = FAMILIES[family](X, Xk, y, cfg, seed=seed)

    offset = int(_cfg(cfg, "ko_offset"))
    sel, tau = _select(W, alpha, offset)

    out = {"method": f"MX-Knockoff-{family}",
           "family": family,
           "selected": sel, "tau": tau, "R": int(sel.size),
           "alpha": float(alpha), "ko_offset": offset,
           "W": W,
           "seconds": round(time.perf_counter() - t0, 3)}
    out.update(info_k)
    out.update(info_s)
    return out


