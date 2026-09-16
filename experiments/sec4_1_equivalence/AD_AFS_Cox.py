"""
AD_AFS_Cox — Augmented-Dummy AFS with a Cox proportional-hazards link.

Explicit-dummy counterpart of the C++ ``VD_AFS_Cox`` (see ``src/vd_afs_cox.cpp``).
Same relaxed forward-stepwise structure as ``AD_AFS`` / ``AD_AFS_Logistic`` /
``AD_AFS_Poisson``, but for survival / time-to-event data:

  - the response ``y`` carries the non-negative event/censoring TIMES; the
    censoring indicator ``delta`` (1 = event observed, 0 = right-censored) is
    supplied separately;
  - there is NO intercept — any constant added to eta cancels inside the
    risk-set ratio, so the baseline hazard already plays the intercept's role;
  - the selection score is the martingale residual
        s_i = delta_i - exp(eta_i) * H0_hat(t_i),
    with H0_hat the Breslow (or Efron) baseline cumulative hazard. Under Breslow
    ties the martingale residuals sum to EXACTLY zero, so the score lives in the
    centered subspace H by construction (a tiny fp mean is removed defensively);
  - a candidate column is scored by  corr(j) = | <x_j, s> |  (single score, so
    no across-class aggregation — unlike the multinomial solver);
  - the OLS solve is replaced by a Cox Newton-Raphson refit on the active set
    (gradient = score of the partial log-likelihood, Hessian = observed
    information, both accumulated over risk sets), and coefficients are damped
    by ``rho``:
        beta <- (1-rho)*beta,  beta[A] += rho*nu
    At rho=1 this is GLM-OMP (full Newton refit each step); rho<1 damps it.

As in the C++ solver there is no VD basis: dummies are explicit columns of
``[X | D]`` and the forward step is a plain argmax over |corr|, like
``AD_AFS_Poisson``.

Interface mirrors ``AD_AFS`` / ``AD_AFS_Logistic`` / ``AD_AFS_Poisson`` so the
same T-Rex FDR wrapper drives it (the only extra constructor argument is the
censoring indicator ``delta``):
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

_ETA_CLAMP = 30.0  # matches VD_AFS_Cox::kEtaClamp


class AD_AFS_Cox:
    """
    Cox Adaptive Forward Stepwise on an augmented design ``[X | D]``.

    Parameters
    ----------
    X : (n, p_real + num_dummies) array
        Augmented design (columns centered / unit-L2, as in the Gaussian pipeline).
    y : (n,) array
        Non-negative event/censoring TIMES.
    delta : (n,) array
        {0,1} censoring indicator (1 = event observed). If ``None`` all samples
        are treated as events (no censoring), mirroring the C++ fallback.
    num_dummies : int
        Number of trailing dummy columns.
    rho : float in (0, 1]
        Blend / relaxation factor. rho=1 -> GLM-OMP.
    efron : bool
        False = Breslow ties (default), True = Efron ties.
    max_steps : int, optional
        Cap on forward-stepwise moves. Defaults to ``p_total + n``.
    newton_max_iter, newton_tol : Newton controls.
    normalize, eps, verbose : as in AD_AFS.
    """

    def __init__(self, X, y, delta, num_dummies, *,
                 rho=1.0,
                 efron=False,
                 max_steps=None,
                 newton_max_iter=25,
                 newton_tol=1e-8,
                 normalize=False,
                 eps=1e-12,
                 verbose=False):

        # --- logging ---
        self.logger = logging.getLogger(f"{__name__}.AD_AFS_Cox")
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
        self.efron = bool(efron)
        self.max_steps = int(max_steps) if max_steps is not None else (self.p_total + self.n)
        self.newton_max_iter = int(newton_max_iter)
        self.newton_tol = float(newton_tol)
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

        # --- Cox state (mirrors init_cox_) ---
        self.time = self.y
        if delta is None:
            self.delta = np.ones(self.n)
        else:
            self.delta = (np.asarray(delta, dtype=float) > 0.5).astype(float)
        self.n_events = int(round(self.delta.sum()))

        # Sample indices sorted by DESCENDING time (once; times never change).
        self.ord_desc = np.argsort(-self.time, kind="stable")

        # Null model: eta = 0, w = 1. Score = null martingale residuals.
        self.eta = np.zeros(self.n)
        self.w = np.ones(self.n)
        self._update_score()

        # --- coefficients / active set ---
        self.beta = np.zeros(self.p_total)
        self.actives = []
        self._actives_set = set()
        self.Xp_A = np.empty((self.n, 0), dtype=float, order="F")
        self._nu = np.empty(0)
        self._newton_stale = True

        self.corr = self.X.T @ self.score              # selection correlation

        # dummy bookkeeping
        self.selected_dummies = []
        self.selected_num_dummies = 0
        self.beta_dict = {}
        self.step = 0

        self.beta_path = [(self.beta / self.normx).copy()]

    # =====================================================================
    # Cox helpers
    # =====================================================================
    def _compute_score(self, eta):
        """Martingale-residual score for a linear predictor ``eta`` (Breslow/Efron).

        Mirrors ``compute_score_``: walk samples in descending time, group ties,
        accumulate the running risk-set sum SR0, form per-distinct-time A_tau /
        B_tau, then H0(t_i) = sum_{tau <= t_i} A_tau and
            s_i = delta_i - w_i * ( H0(t_i) - 1{event at t_i} * (A - B) ).
        """
        n = self.n
        w = np.exp(np.clip(eta, -_ETA_CLAMP, _ETA_CLAMP))

        dt_A, dt_B = [], []            # A_tau, B_tau per distinct time (descending)
        grp_of = np.empty(n, dtype=int)

        SR0 = 0.0
        g = 0
        idx = 0
        ordv = self.ord_desc
        N = n
        while idx < N:
            tau = self.time[ordv[idx]]
            j = idx
            SD0 = 0.0
            d = 0
            # add every member of this tie group to the risk set
            while j < N and self.time[ordv[j]] == tau:
                s = ordv[j]
                SR0 += w[s]
                grp_of[s] = g
                if self.delta[s] > 0.5:
                    SD0 += w[s]
                    d += 1
                j += 1
            A = 0.0
            B = 0.0
            if d > 0:
                if not self.efron:
                    A = B = d / SR0                                # Breslow
                else:
                    for l in range(d):                             # Efron
                        frac = l / d
                        denom = SR0 - frac * SD0
                        inv = (1.0 / denom) if denom > 0.0 else 0.0
                        A += inv
                        B += (1.0 - frac) * inv
            dt_A.append(A)
            dt_B.append(B)
            idx = j
            g += 1
        G = g

        # Cumulative baseline hazard, accumulated in ASCENDING time (groups were
        # stored descending, so iterate from the last group to the first).
        cumA = np.empty(G)
        run = 0.0
        for gg in range(G - 1, -1, -1):
            run += dt_A[gg]
            cumA[gg] = run

        dt_A = np.asarray(dt_A)
        dt_B = np.asarray(dt_B)
        H0 = cumA[grp_of].copy()
        ev = self.delta > 0.5
        H0[ev] -= (dt_A[grp_of[ev]] - dt_B[grp_of[ev]])            # own-time swap
        return self.delta - w * H0

    def _update_score(self):
        """Refresh w / centered martingale score from the current eta."""
        self.w = np.exp(np.clip(self.eta, -_ETA_CLAMP, _ETA_CLAMP))
        s = self._compute_score(self.eta)
        s = s - s.mean()                # sums to zero algebraically; be defensive
        self.score = s
        self.residuals = s
        self.mu = self.w

    def _cox_grad_hess(self, A, eta):
        """Cox gradient g (k) and observed information H (k x k) at ``eta``.

        Mirrors ``cox_grad_hess_``: same risk-set accumulation as the score,
        extended to first (SR1) and second (SR2) moments of the rows of A.
        """
        n = self.n
        k = A.shape[1]
        g = np.zeros(k)
        H = np.zeros((k, k))
        if k == 0:
            return g, H

        w = np.exp(np.clip(eta, -_ETA_CLAMP, _ETA_CLAMP))
        ordv = self.ord_desc
        N = n

        SR0 = 0.0
        SR1 = np.zeros(k)
        SR2 = np.zeros((k, k))

        idx = 0
        while idx < N:
            tau = self.time[ordv[idx]]
            j = idx
            SD0 = 0.0
            SD1 = np.zeros(k)
            SD2 = np.zeros((k, k))
            sumD_a = np.zeros(k)
            d = 0
            while j < N and self.time[ordv[j]] == tau:
                s = ordv[j]
                a = A[s, :]
                ws = w[s]
                SR0 += ws
                SR1 += ws * a
                SR2 += ws * np.outer(a, a)
                if self.delta[s] > 0.5:
                    SD0 += ws
                    SD1 += ws * a
                    SD2 += ws * np.outer(a, a)
                    sumD_a += a
                    d += 1
                j += 1

            if d > 0:
                g += sumD_a
                if not self.efron:                                 # Breslow
                    inv = 1.0 / SR0
                    mean1 = SR1 * inv
                    g -= d * mean1
                    H += d * (SR2 * inv - np.outer(mean1, mean1))
                else:                                              # Efron
                    for l in range(d):
                        frac = l / d
                        denom = SR0 - frac * SD0
                        if denom <= 0.0:
                            continue
                        inv = 1.0 / denom
                        num1 = SR1 - frac * SD1
                        num2 = SR2 - frac * SD2
                        mean1 = num1 * inv
                        g -= mean1
                        H += num2 * inv - np.outer(mean1, mean1)
            idx = j
        return g, H

    def _is_dummy_col(self, j):
        return j >= self.p_real

    def _add_active(self, j):
        self.Xp_A = np.column_stack((self.Xp_A, self.X[:, j])) if self.Xp_A.shape[1] \
            else self.X[:, j].reshape(self.n, 1).copy()
        self.actives.append(int(j))
        self._actives_set.add(int(j))
        if self.entry_step[j] < 0:
            self.entry_step[j] = self.step
        self._newton_stale = True
        if self._is_dummy_col(j):
            self.selected_num_dummies += 1
            self.selected_dummies.append(int(j))
            self.beta_dict[self.selected_num_dummies] = None  # filled after blend

    def _newton_solve(self):
        """Cox Newton-Raphson refit on the active set (no intercept)."""
        k = len(self.actives)
        if k == 0:
            self._nu = np.empty(0)
            self._newton_stale = False
            return

        A = self.Xp_A
        # warm start from current blended coefficients
        theta = self.beta[np.asarray(self.actives, dtype=int)].copy()

        for _ in range(self.newton_max_iter):
            eta_a = A @ theta
            g, H = self._cox_grad_hess(A, eta_a)
            H[np.diag_indices_from(H)] += self.eps        # ridge
            try:
                step = np.linalg.solve(H, g)              # Newton: theta += H^{-1} g
            except np.linalg.LinAlgError:
                break
            theta = theta + step
            if float(np.sum(step ** 2)) < self.newton_tol ** 2 * (1.0 + float(np.sum(theta ** 2))):
                break

        self._nu = theta
        self._newton_stale = False

    def _afs_blend(self):
        """One damped Cox-Newton blend (mirrors ``afs_blend_``); no intercept."""
        k = len(self.actives)
        if k == 0:
            return
        if self._newton_stale:
            self._newton_solve()

        self.beta *= (1.0 - self.rho)
        actives_np = np.asarray(self.actives, dtype=int)
        self.beta[actives_np] += self.rho * self._nu

        self.eta = self.Xp_A @ self.beta[actives_np]      # no intercept
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
