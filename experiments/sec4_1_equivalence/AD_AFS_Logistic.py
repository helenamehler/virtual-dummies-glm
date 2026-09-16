"""
AD_AFS_Logistic — Augmented-Dummy AFS with a logistic link (explicit dummies).

Explicit-dummy counterpart of the C++ ``VD_AFS_Logistic`` (see
``src/vd_afs_logistic.cpp``). Same relaxed forward-stepwise structure as
``AD_AFS``, but:

  - selection is driven by the *score* direction  s = y_binary - sigmoid(eta),
    (centered), so  corr = X^T s;
  - the OLS solve is replaced by an IRLS (Newton) refit of a logistic GLM on the
    always-on intercept plus the active columns of ``[X | D]``;
  - coefficients and intercept are damped by ``rho``:
        b0   <- (1-rho)*b0   + rho*b0_irls
        beta <- (1-rho)*beta,  beta[A] += rho*nu
    At rho=1 this is GLM-OMP (full IRLS refit each step); rho<1 damps the update.

The binary labels are reconstructed from the sign of the (centered) response
``y`` that the T-Rex pipeline passes in, exactly as ``init_logistic_`` does in the
C++ code: ``y_binary = 1[y > 0]``.

Interface mirrors ``AD_LARS`` / ``AD_AFS`` so the same T-Rex FDR wrapper drives it:
  - warm-start ``run(T, stop="dummies")``
  - ``beta_dict[t]["beta"]`` snapshot when the t-th dummy entered
  - ``beta_path`` per-step snapshots (full length p_total; reals are ``[:p_real]``)

Requires only numpy (+ scipy is not needed here; solves use numpy.linalg).
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.

from __future__ import annotations

import logging
import sys

import numpy as np


def _sigmoid(eta):
    # clip to avoid exp overflow under (quasi-)separation; sigmoid saturates anyway
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


class AD_AFS_Logistic:
    """
    Logistic Adaptive Forward Stepwise on an augmented design ``[X | D]``.

    Parameters
    ----------
    X : (n, p_real + num_dummies) array
        Augmented design (columns centered / unit-L2, as in the Gaussian pipeline).
    y : (n,) array
        Centered response; binary labels are recovered as ``1[y > 0]``.
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
        self.logger = logging.getLogger(f"{__name__}.AD_AFS_Logistic")
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

        # --- logistic state (mirrors init_logistic_) ---
        self.y_binary = (self.y > 0.0).astype(float)
        pbar = float(np.clip(self.y_binary.mean(), 1e-6, 1.0 - 1e-6))
        self.b0 = np.log(pbar / (1.0 - pbar))     # logit(pbar)
        self.b0_irls = self.b0
        self.eta = np.full(self.n, self.b0)
        self._update_score()                       # sets prob, score, mu, residuals

        # --- coefficients / active set ---
        self.beta = np.zeros(self.p_total)
        self.actives = []
        self._actives_set = set()
        self.Xp_A = np.empty((self.n, 0), dtype=float, order="F")
        self._nu = np.empty(0)
        self._irls_stale = True

        self.corr = self.X.T @ self.score          # selection correlation

        # dummy bookkeeping
        self.selected_dummies = []
        self.selected_num_dummies = 0
        self.beta_dict = {}
        self.step = 0

        self.beta_path = [(self.beta / self.normx).copy()]

    # =====================================================================
    # Logistic helpers
    # =====================================================================
    def _update_score(self):
        """Recompute prob / (centered) score / mu / residuals from current eta."""
        self.prob = _sigmoid(self.eta)
        s = self.y_binary - self.prob
        s = s - s.mean()
        self.score = s
        self.mu = self.prob
        self.residuals = s

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
        """Newton/IRLS refit of [1 | X_active] on y_binary (mirrors irls_solve_)."""
        k = len(self.actives)
        Xa = np.empty((self.n, k + 1))
        Xa[:, 0] = 1.0
        if k:
            Xa[:, 1:] = self.Xp_A
        # warm start from current parameters
        theta = np.empty(k + 1)
        theta[0] = self.b0
        if k:
            theta[1:] = self.beta[np.asarray(self.actives, dtype=int)]

        lo, hi = 1e-10, 1.0 - 1e-10
        for _ in range(self.irls_max_iter):
            eta_a = Xa @ theta
            p_a = np.clip(_sigmoid(eta_a), lo, hi)
            w = p_a * (1.0 - p_a)
            z = eta_a + (self.y_binary - p_a) / w
            sw = np.sqrt(w)
            Xw = Xa * sw[:, None]
            zw = z * sw
            G = Xw.T @ Xw
            G[np.diag_indices_from(G)] += self.eps
            rhs = Xw.T @ zw
            try:
                theta_new = np.linalg.solve(G, rhs)
            except np.linalg.LinAlgError:
                break
            delta = float(np.sum((theta_new - theta) ** 2))
            theta = theta_new
            if delta < self.irls_tol ** 2 * (1.0 + float(np.sum(theta ** 2))):
                break

        self.b0_irls = float(theta[0])
        self._nu = theta[1:].copy()
        self._irls_stale = False

    def _afs_blend(self):
        """One damped IRLS blend (mirrors VD_AFS_Logistic::afs_blend_)."""
        k = len(self.actives)
        if k == 0:
            return
        if self._irls_stale:
            self._irls_solve()

        self.b0 = (1.0 - self.rho) * self.b0 + self.rho * self.b0_irls
        self.beta *= (1.0 - self.rho)
        actives_np = np.asarray(self.actives, dtype=int)
        self.beta[actives_np] += self.rho * self._nu

        self.eta = self.Xp_A @ self.beta[actives_np] + self.b0
        self._update_score()
        self.corr = self.X.T @ self.score

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
