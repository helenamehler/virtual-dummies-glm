"""
AD_AFS_Multinomial — Augmented-Dummy AFS with a multinomial (softmax) link.

Explicit-dummy counterpart of the C++ ``VD_AFS_Multinomial`` (see
``src/vd_afs_multinomial.cpp``). Same relaxed forward-stepwise structure as
``AD_AFS`` / ``AD_AFS_Logistic`` / ``AD_AFS_Poisson``, but with a reference-
category softmax score and a multinomial IRLS refit:

  - class 0 is the reference; for ``C`` classes there are ``Cm = C-1`` non-
    reference score directions  s^(c) = y^(c) - p^(c),  p = softmax(eta), each
    an n-vector projected back to the centered subspace H;
  - a candidate column is scored by the L2 norm across ALL C classes of its
    per-class inner products, computed from the C-1 tracked ones via the
    zero-sum identity (see ``_corr_from_score``). The resulting rank statistic
    does not depend on which class is held out, exactly the aggregation used by
    ``full_corr_refresh_multi_`` in the C++ code;
  - the OLS solve is replaced by a reference-category multinomial Newton/IRLS
    refit of a softmax GLM on the always-on intercept plus the active columns of
    ``[X | D]`` (flattened class-major Fisher system, see ``irls_solve_``);
  - coefficients and per-class intercepts are damped by ``rho``:
        b0   <- (1-rho)*b0   + rho*b0_irls
        Beta <- (1-rho)*Beta,  Beta[A] += rho*Nu
    At rho=1 this is GLM-OMP (full IRLS refit each step); rho<1 damps the update.

Unlike the C++ VD selector, there is no VD basis here: dummies are explicit
columns of ``[X | D]`` and the forward step is a plain argmax over the (multi-
class-aggregated) correlation, exactly like ``AD_AFS_Poisson``.

The labels are the RAW integer class labels 0..C-1 that the T-Rex pipeline
passes in, exactly as ``init_multinomial_`` does in the C++ code. C is inferred
from ``max(y)+1``. The null intercepts are the class-marginal log-odds against
the reference,  b0_c = log(n_c / n_0).

Because ``beta`` in the returned path must stay a single scalar per feature (the
FDR wrapper reads ``|beta[:p]| > EPS``), we fill it with the ACROSS-CLASS L2 norm
of the per-class coefficients — nonzero iff the feature ever entered the active
set, matching ``beta_real()`` in the C++ solver.

Interface mirrors ``AD_AFS`` / ``AD_AFS_Logistic`` / ``AD_AFS_Poisson`` so the
same T-Rex FDR wrapper drives it:
  - warm-start ``run(T, stop="dummies")``
  - ``beta_dict[t]["beta"]`` snapshot when the t-th dummy entered
  - ``beta_path`` per-step snapshots (full length p_total; reals are ``[:p_real]``)

Requires only numpy (solves use numpy.linalg).
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.

from __future__ import annotations

import logging
import sys

import numpy as np

_ETA_CLAMP = 30.0  # matches VD_AFS_Multinomial::kEtaClamp


def _softmax_ref(eta):
    """Reference-category softmax. ``eta`` is (n, Cm); class 0 has eta == 0.

    p_ic = exp(eta_ic) / (1 + sum_c' exp(eta_ic')),  c = 1..Cm.
    """
    e = np.exp(np.clip(eta, -_ETA_CLAMP, _ETA_CLAMP))
    denom = 1.0 + e.sum(axis=1, keepdims=True)  # + exp(0) of the reference class
    return e / denom


class AD_AFS_Multinomial:
    """
    Multinomial Adaptive Forward Stepwise on an augmented design ``[X | D]``.

    Parameters
    ----------
    X : (n, p_real + num_dummies) array
        Augmented design (columns centered / unit-L2, as in the Gaussian pipeline).
    y : (n,) array
        RAW integer class labels in 0..C-1 (not centered). C = max(y)+1.
    num_dummies : int
        Number of trailing dummy columns.
    rho : float in (0, 1]
        Blend / relaxation factor. rho=1 -> GLM-OMP.
    max_steps : int, optional
        Cap on forward-stepwise moves (features may be re-selected when rho<1).
        Defaults to ``p_total + n``.
    irls_max_iter, irls_tol : IRLS Newton controls.
    normalize, eps, verbose : as in AD_AFS.
    """

    def __init__(self, X, y, num_dummies, *,
                 rho=1.0,
                 max_steps=None,
                 irls_max_iter=25,
                 irls_tol=1e-8,
                 normalize=False,
                 eps=1e-12,
                 verbose=False):

        # --- logging ---
        self.logger = logging.getLogger(f"{__name__}.AD_AFS_Multinomial")
        level = logging.DEBUG if verbose else logging.WARNING
        self.logger.setLevel(level)
        for h in list(self.logger.handlers):
            if isinstance(h, logging.StreamHandler):
                self.logger.removeHandler(h)
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.logger.addHandler(handler)
        self.logger.propagate = False

        # --- data ---
        self.X = np.asarray(X, dtype=float, order="F")
        self.y = np.asarray(y, dtype=float).copy()
        self.n, self.p_total = self.X.shape

        self.num_dummies = int(num_dummies)
        if not (0 <= self.num_dummies <= self.p_total):
            raise ValueError("num_dummies must be between 0 and p_total.")
        self.p_real = self.p_total - self.num_dummies

        self.rho = float(max(1e-6, min(rho, 1.0)))
        self.max_steps = int(max_steps) if max_steps is not None else (self.p_total + self.n)
        self.irls_max_iter = int(irls_max_iter)
        self.irls_tol = float(irls_tol)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        self.normx = np.ones(self.p_total, dtype=float)
        if self.normalize:
            self.normx = np.linalg.norm(self.X, axis=0)
            safe = self.normx > self.eps * np.sqrt(self.n)
            self.normx[~safe] = self.eps * np.sqrt(self.n)
            self.X = self.X / self.normx

        # per-column entry step
        self.entry_step = -np.ones(self.p_total, dtype=int)

        # --- multinomial state (mirrors init_multinomial_) ---
        lab = np.clip(np.rint(self.y).astype(int), 0, None)
        self.C = int(max(2, lab.max() + 1))
        self.Cm = self.C - 1

        # {0,1} indicator columns for the non-reference classes 1..C-1.
        self.Y_ind = np.zeros((self.n, self.Cm))
        for c in range(1, self.C):
            self.Y_ind[:, c - 1] = (lab == c).astype(float)

        # Null intercepts: b0_c = log(p_c / p_0), class marginals.
        counts = self.Y_ind.sum(axis=0)               # n_c for c = 1..C-1
        n0 = max(self.n - counts.sum(), 1e-8)          # reference-class count
        p0 = max(n0 / self.n, 1e-8)
        pc = np.maximum(counts / self.n, 1e-8)
        self.b0 = np.log(pc / p0)                      # (Cm,)
        self.b0_irls = self.b0.copy()
        self.eta = np.tile(self.b0, (self.n, 1))       # (n, Cm) constant rows
        self._update_score()                           # sets prob, centered score

        # --- coefficients / active set ---
        # Per-class coefficients for every column of [X | D] (reals + dummies).
        self.beta_multi = np.zeros((self.p_total, self.Cm))
        self.beta = np.zeros(self.p_total)             # per-feature L2 norm (path scalar)
        self.actives = []
        self._actives_set = set()
        self.Xp_A = np.empty((self.n, 0), dtype=float, order="F")
        self._Nu = np.empty((0, self.Cm))
        self._irls_stale = True

        self.corr = self._corr_from_score()            # selection correlation (>= 0)

        # dummy bookkeeping
        self.selected_dummies = []
        self.selected_num_dummies = 0
        self.beta_dict = {}
        self.step = 0

        self.beta_path = [(self.beta / self.normx).copy()]

    # =====================================================================
    # Multinomial helpers
    # =====================================================================
    def _update_score(self):
        """Recompute prob / (centered) per-class score from current eta.

        Each class column of the raw score y^(c) - p^(c) is projected back to the
        centered subspace H (mean removed), mirroring ``update_score_()``: with
        per-class intercepts at the MLE the column sums are exactly zero, but
        under AFS blending (rho<1) they are only approximately zero.
        """
        self.prob = _softmax_ref(self.eta)             # (n, Cm)
        s = self.Y_ind - self.prob
        s = s - s.mean(axis=0, keepdims=True)          # center each class column
        self.score = s
        self.mu = self.prob
        self.residuals = s[:, 0]                        # loose base-class alias

    def _corr_from_score(self):
        """Reference-invariant across-class score norm.

        The full C-class score vector of candidate j,
            g_j = ( <x_j, s^(0)>, ..., <x_j, s^(C-1)> ),
        sums to zero, because sum_c ( y_i^(c) - p_i^(c) ) = 0 for every i. Only
        the Cm = C-1 non-reference entries are stored, so the reference entry is
            g_j^(0) = - sum_{c>=1} g_j^(c)
        and the full norm follows without refitting anything:
            ||g_j||^2 = ||G_j||^2 + ( sum_c G_jc )^2 .
        Relabeling the classes permutes the entries of g_j, so ||g_j||_2 is
        unchanged. The truncated norm ||G_j||_2 is not: it depends on which
        class was held out.
        """
        G = self.X.T @ self.score                       # (p_total, Cm)
        return np.sqrt(np.einsum("jc,jc->j", G, G) + G.sum(axis=1) ** 2)

    def _is_dummy_col(self, j):
        return j >= self.p_real

    def _add_active(self, j):
        self.Xp_A = np.column_stack((self.Xp_A, self.X[:, j])) if self.Xp_A.shape[1] \
            else self.X[:, j].reshape(self.n, 1).copy()
        self.actives.append(int(j))
        self._actives_set.add(int(j))
        if self.entry_step[j] < 0:
            self.entry_step[j] = self.step
        self._irls_stale = True
        if self._is_dummy_col(j):
            self.selected_num_dummies += 1
            self.selected_dummies.append(int(j))
            self.beta_dict[self.selected_num_dummies] = None  # filled after blend

    def _irls_solve(self):
        """Reference-category multinomial Newton on [1 | X_active].

        Mirrors ``irls_solve_``: parameter matrix Theta is (kk x Cm) with
        kk = k+1 (intercept + active columns), flattened class-major into a dense
        M x M Fisher system (M = kk*Cm),
            Hessian block (c, c') = Xa^T diag( p_c ([c==c'] - p_c') ) Xa,
            gradient block  (c)   = Xa^T ( y^(c) - p^(c) ).
        """
        k = len(self.actives)
        kk = k + 1
        M = kk * self.Cm

        Xa = np.empty((self.n, kk))
        Xa[:, 0] = 1.0
        if k:
            Xa[:, 1:] = self.Xp_A

        # Warm start Theta from current blended values.
        Theta = np.empty((kk, self.Cm))
        Theta[0, :] = self.b0
        if k:
            Theta[1:, :] = self.beta_multi[np.asarray(self.actives, dtype=int), :]

        for _ in range(self.irls_max_iter):
            eta_a = Xa @ Theta                          # (n, Cm)
            Pa = _softmax_ref(eta_a)                    # (n, Cm)

            g = np.zeros(M)
            H = np.zeros((M, M))
            for c in range(self.Cm):
                resid = self.Y_ind[:, c] - Pa[:, c]
                g[c * kk:(c + 1) * kk] = Xa.T @ resid
                for cp in range(c, self.Cm):
                    w = Pa[:, c] * ((1.0 if c == cp else 0.0) - Pa[:, cp])
                    B = Xa.T @ (Xa * w[:, None])         # (kk, kk)
                    H[c * kk:(c + 1) * kk, cp * kk:(cp + 1) * kk] = B
                    if cp != c:
                        H[cp * kk:(cp + 1) * kk, c * kk:(c + 1) * kk] = B.T

            H[np.diag_indices_from(H)] += self.eps       # ridge
            try:
                delta = np.linalg.solve(H, g)            # Newton step
            except np.linalg.LinAlgError:
                break
            step_mat = delta.reshape(self.Cm, kk).T      # (kk, Cm), class-major
            Theta = Theta + step_mat
            dn = float(np.sum(delta ** 2))
            tn = float(np.sum(Theta ** 2))
            if dn < self.irls_tol ** 2 * (1.0 + tn):
                break

        self.b0_irls = Theta[0, :].copy()
        self._Nu = Theta[1:, :].copy() if k else np.empty((0, self.Cm))
        self._irls_stale = False

    def _afs_blend(self):
        """One damped multinomial-IRLS blend (mirrors ``afs_blend_``)."""
        k = len(self.actives)
        if k == 0:
            return
        if self._irls_stale:
            self._irls_solve()

        self.b0 = (1.0 - self.rho) * self.b0 + self.rho * self.b0_irls
        self.beta_multi *= (1.0 - self.rho)
        actives_np = np.asarray(self.actives, dtype=int)
        self.beta_multi[actives_np, :] += self.rho * self._Nu

        self.eta = self.Xp_A @ self.beta_multi[actives_np, :] + self.b0
        self._update_score()
        self.corr = self._corr_from_score()
        # Per-feature representative magnitude, on the sum-to-zero (symmetric)
        # coefficient representation so it is free of the reference class too:
        # with b^(0) = 0 and m_j = mean_c b_j^(c) over ALL C classes,
        #   || b_j - m_j 1 ||^2 = ||B_j||^2 - ( sum_c B_jc )^2 / C .
        # Zero iff the feature never entered the active set, as before.
        self.beta = np.sqrt(np.maximum(
            np.einsum("jc,jc->j", self.beta_multi, self.beta_multi)
            - self.beta_multi.sum(axis=1) ** 2 / self.C, 0.0))

    # =====================================================================
    # Main loop (warm-start capable)
    # =====================================================================
    def run(self, T=1, *, stop="dummies"):
        T = int(T)
        if T < 0:
            raise ValueError("T must be >= 0.")
        stop = str(stop).lower()
        if stop not in {"dummies", "steps"}:
            raise ValueError("stop must be 'dummies' or 'steps'.")

        if stop == "dummies" and self.selected_num_dummies >= T:
            return np.array(self.beta_path)
        if stop == "steps" and self.step >= T:
            return np.array(self.beta_path)

        while self.step < self.max_steps:
            if stop == "steps" and self.step >= T:
                break

            j = int(np.argmax(np.abs(self.corr)))
            if abs(float(self.corr[j])) < 100.0 * self.eps:
                break

            prev_dummies = self.selected_num_dummies
            is_new = j not in self._actives_set
            if is_new:
                self._add_active(j)

            self._afs_blend()
            snap = (self.beta / self.normx).copy()
            self.beta_path.append(snap)

            if is_new and self._is_dummy_col(j):
                self.beta_dict[self.selected_num_dummies] = {"beta": snap.copy()}

            self.step += 1

            if (stop == "dummies"
                    and self.selected_num_dummies > prev_dummies
                    and self.selected_num_dummies >= T):
                break

        return np.array(self.beta_path)
