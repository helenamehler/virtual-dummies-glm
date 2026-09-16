"""
Sec. 4.1, Fig. 1, Bernoulli column. FDP and TPP of AD-AFS-Log, VD-AFS-Log,
VD-AFS-Gauss (misspecified linear model) and model-X knockoffs against the SNR.

SNR = Var(mu) / E[mu (1 - mu)], see snr_glm.py. The linear predictor is scaled
by bisection to hit the target exactly, with the intercept fixed so that the
prevalence is PREVALENCE; ``snr_exact`` logs the realised value. The knockoff
arm needs knockpy and skglm and is skipped without them.
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
from __future__ import annotations

import os, json
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# numexpr caps itself at 64 threads and warns loudly when the
# allocation is wider; pin it like the rest.
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# skglm's solver is numba-compiled; one worker per core means one thread each.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
import pandas as pd

from vd_selectors import (
    TRexSelector, TRexOptions, SolverType, CalibMode, VDDummyLaw,
)
from AD_AFS_Logistic import AD_AFS_Logistic

# Model-X knockoff arm. Optional on purpose: the cluster environment may not
# carry knockpy and skglm, and a missing competitor must not take the sweep
# down with it.
try:
    import knockoff_glm as kog
    KNOCKOFF_OK, KNOCKOFF_WHY = True, ""
except Exception as _e:                                   # pragma: no cover
    kog, KNOCKOFF_OK, KNOCKOFF_WHY = None, False, str(_e)

# SKIP_KO=1 drops the knockoff arm, the expensive part of the sweep.
if os.environ.get("SKIP_KO"):
    KNOCKOFF_OK, KNOCKOFF_WHY = False, "disabled via SKIP_KO"

KO_CFG = {}          # overrides on top of knockoff_glm.DEFAULT_CFG

# Unified SNR axis (see snr_glm.py): the response-scale ratio
# SNR = Var{E(y|x)} / E{Var(y|x)}, calibrated per replicate.
from snr_glm import snr_bernoulli, unit_variance, calibrate


EPS = 1e-12

# Prevalence under the null. It fixes the number of EVENTS, which is what
# actually limits power here. Diagnostic (p=1000, s=10, rank of the s active columns under the
# step-1 score |<x_j, y - ybar>|; "in Top10" out of 10, averaged over 30 designs):
#
#   n=300, pi0=0.1 (30 events):  SNR 0.1->4 gives 1.1  2.9  3.9  5.5  5.7  5.8
#   n=300, pi0=0.3 (90 events):                  1.4  2.8  4.9  6.4  7.4  8.2
#   n=300, pi0=0.5 (150 events):                 1.6  3.2  5.1  6.8  7.7  8.5
#   n=1000, pi0=0.1 (100 events):                5.0  7.8  9.1  9.4  9.8  9.9
#
# At n=300 with pi0=0.1 the ranking saturates at ~5.8/10 from SNR=1 on: once the
# fitted probabilities saturate, extra signal carries no extra information about
# individual columns, so TPP goes FLAT. That is a property of the design, not of
# the selectors. Either raise the prevalence or raise n.
PREVALENCE = 0.5


# ================================================================
# Worker count
#
# NEVER use n_jobs=-1 under Slurm: joblib/loky then sizes the pool from what it
# believes the machine has, which — depending on whether Slurm applies a cpuset —
# can be the whole node instead of the allocated cores, while the memory cgroup
# still only grants ``-c`` * ``--mem-per-cpu``. The AD baseline costs ~0.8 GB per
# worker at L=10 (K persistent [X|D] copies, ~n*p*(1+L)*8 bytes each), so an
# oversubscribed pool walks straight into OUT_OF_MEMORY. Read the allocation.
# ================================================================
def slurm_n_jobs(default=-1, cap=None):
    """Worker count from the Slurm allocation, optionally capped.

    ``cap`` trades throughput for headroom: the memory cgroup is
    ``-c`` * ``--mem-per-cpu`` regardless of how many workers run, so fewer
    workers means MORE memory per worker. With -c 96 / --mem-per-cpu 3800 the
    budget is 364.8 GB, i.e. 3.8 GB per worker at 96 and 7.3 GB at 50.
    """
    n = default
    for var in ("SLURM_CPUS_PER_TASK", "LOKY_MAX_CPU_COUNT"):
        v = os.environ.get(var)
        if v and v.isdigit() and int(v) > 0:
            n = int(v)
            break
    if cap is not None and n > 0:
        n = min(n, int(cap))
    return n


def ad_arm_mem_estimate_gb(n, p, L, K, n_jobs):
    """Rough peak RSS of the AD arm: n_jobs * K * n * p * (1+L) * 8 bytes."""
    return n_jobs * K * n * p * (1 + L) * 8 / 1e9


# ================================================================
# tqdm + joblib integration
# ================================================================
@contextmanager
def tqdm_joblib(tqdm_object):
    import joblib
    class _Cb(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *a, **kw):
            try: tqdm_object.update(n=self.batch_size)
            except Exception: pass
            return super().__call__(*a, **kw)
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Cb
    try: yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()


# ================================================================
# Utilities
# ================================================================
def center_unitL2(X, eps=EPS, fortran=False):
    X = np.asarray(X, float)
    X -= X.mean(axis=0, keepdims=True)
    X /= np.linalg.norm(X, axis=0, keepdims=True).clip(min=eps)
    return np.asfortranarray(X) if fortran else X

def fdr_tpp(selected, truth):
    sel = set(map(int, np.asarray(selected, int).ravel()))
    tru = set(map(int, np.asarray(truth, int).ravel()))
    tp, fp, R, P = len(sel & tru), len(sel - tru), len(sel), len(tru)
    return (fp / R if R else 0.0), (tp / P if P else 0.0)


def make_problem(n=1000, p=500, s=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    X = center_unitL2(rng.standard_normal((n, p)), fortran=True)
    support = np.sort(rng.choice(p, size=s, replace=False))
    beta_base = np.zeros(p)
    beta_base[support] = rng.choice([-1.0, 1.0], size=s)  # unit sign pattern
    return X, beta_base, support

def _sigmoid(z):
    """Numerisch stabile Logistik-Funktion."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _solve_intercept(eta, target, lo=-50.0, hi=50.0, tol=1e-10, max_iter=200):
    """b0 s.t. mean(sigmoid(b0 + eta)) == target. Monoton -> Bisektion."""
    t = float(target)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if float(_sigmoid(mid + eta).mean()) < t:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def sample_y_snr(X, beta_base, snr, rng, prevalence=None):
    """Centered 0/1 response at target SNR = Var(mu)/E[mu(1-mu)].

    Returns the centered response, the realised SNR and the realised
    prevalence.
    """
    u = unit_variance(X @ beta_base)

    def prob_of(c):
        eta = float(c) * u
        b0 = 0.0 if prevalence is None else _solve_intercept(eta, prevalence)
        return _sigmoid(b0 + eta)

    prob = prob_of(calibrate(float(snr), lambda c: snr_bernoulli(prob_of(c))))
    y = rng.binomial(1, prob).astype(float)
    return y - y.mean(), snr_bernoulli(prob), float(y.mean())


# ================================================================
# T-Rex math helpers (FDP estimator; solver-agnostic)
# ================================================================
def phi_prime(p, T, Ld, phi_T, Phi):
    av = phi_T.sum(axis=0)
    rows = Phi > 0.5
    delta = phi_T[rows].sum(axis=0) if rows.any() else np.zeros(T)
    phi_mod, delta_mod = phi_T.copy(), delta.copy()
    if T > 1:
        phi_mod[:, 1:] -= phi_T[:, :-1]
        delta_mod[1:] -= delta[:-1]
    w = np.zeros(T)
    for t in range(T):
        denom = Ld - t
        if delta_mod[t] > EPS and denom > 0:
            w[t] = 1.0 - (p - av[t]) / (denom * delta_mod[t])
    return phi_mod @ w

def fdp_hat(V, Phi, PhiP):
    out = np.zeros_like(V)
    for i, v in enumerate(V):
        idx = Phi > v
        R = int(idx.sum())
        out[i] = 0.0 if R == 0 else min(1.0, float(np.sum(1.0 - PhiP[idx]) / R))
    return out

def select_vars(p, tFDR, FDP_mat, Phi_mat, V):
    feasible_T = np.where((FDP_mat <= tFDR).any(axis=1))[0]
    if feasible_T.size == 0:
        return np.array([], dtype=int), 0, float(V[-1]), 0
    T_select = int(feasible_T.max())
    R_mat = np.zeros_like(FDP_mat[:T_select+1], dtype=int)
    for t in range(T_select + 1):
        R_mat[t] = (Phi_mat[t][:, None] > V[None, :]).sum(axis=0)
    R_masked = np.where(FDP_mat[:T_select+1] <= tFDR, R_mat, -1)
    max_R = int(R_masked.max())
    locs = np.argwhere(R_masked == max_R)
    locs = locs[np.lexsort((locs[:, 0], locs[:, 1]))]
    t_idx, v_idx = locs[-1]
    v_star = float(V[v_idx])
    return np.flatnonzero(Phi_mat[t_idx] > v_star).astype(int), int(t_idx+1), v_star, max_R


# ================================================================
# AD-AFS-Logistic-T-Rex (Python AD_AFS_Logistic)
# ================================================================
def trex_ad_select(X, y, *, tFDR, K, num_dummies, Tmax, rho, seed):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    V = np.append(np.arange(0.5, 1.0, 1.0 / K), 1.0 - EPS)

    solvers = []
    for _ in range(K):
        D = center_unitL2(rng.standard_normal((n, num_dummies)))
        XD = np.asfortranarray(np.hstack([X, D]))
        tl = AD_AFS_Logistic(XD, np.asarray(y), int(num_dummies), rho=rho,
                             normalize=False, eps=EPS, verbose=False)
        solvers.append(tl)

    FDP_list, Phi_list = [], []

    for T in range(1, Tmax + 1):
        phi_acc = np.zeros(p)
        for tl in solvers:
            tl.run(T=T)
            if T in tl.beta_dict and tl.beta_dict[T] is not None:
                beta_t = tl.beta_dict[T]["beta"]
            else:
                beta_t = tl.beta_path[-1]
            phi_acc += (np.abs(beta_t[:p]) > EPS)
        Phi = phi_acc / K

        Phi_list.append(Phi)
        phi_T_mat = np.column_stack(Phi_list)

        PhiP = phi_prime(p, T, num_dummies, phi_T_mat, Phi)
        FDP = fdp_hat(V, Phi, PhiP)
        FDP_list.append(FDP)

        if FDP[-1] > tFDR:
            break

    FDP_mat = np.vstack(FDP_list)
    Phi_mat_out = np.vstack([ph.reshape(1, -1) for ph in Phi_list])
    return select_vars(p, tFDR, FDP_mat, Phi_mat_out, V)


# ================================================================
# VD-AFS-Logistic-T-Rex (C++ vd_selectors)
# ================================================================
def trex_vd_select(X, y, *, tFDR, K, L_factor, Tmax, rho, seed):
    opt = TRexOptions()
    opt.tFDR = float(tFDR)
    opt.K = int(K)
    opt.L_factor = int(L_factor)
    opt.T_stop = int(Tmax)
    opt.seed = int(seed)
    opt.verbose = False
    opt.solver = SolverType.AFS_Logistic  # type: ignore
    opt.rho = float(rho)
    opt.calib = CalibMode.CalibrateT
    opt.dummy_law = VDDummyLaw.Spherical
    opt.n_threads = 1
    opt.max_stale_strides = 999
    opt.stride_width = 1
    opt.posthoc_mode = False

    sel = TRexSelector(opt)
    res = sel.run(np.asfortranarray(X), np.asarray(y))
    selected = np.asarray(res.selected_var, dtype=int)
    T_star = int(res.T_stop)
    v_star = float(res.v_thresh)
    return selected, T_star, v_star, len(selected)


# ================================================================
# VD-AFS-Gauss-T-Rex (C++ vd_selectors) — Gaussian / OLS AFS baseline
#
# Runs the plain Gaussian (least-squares) AFS T-Rex on the SAME binary response
# (already centered by ``sample_y_snr``), i.e. it treats the 0/1 outcome as a
# linear-Gaussian one — the linear probability model. Same solver family and the
# same ``rho`` as VD-AFS-Log-T-Rex, only the likelihood differs, so the pair is a
# clean apples-to-apples test of whether modelling the logistic link actually
# buys anything over the much cheaper OLS-based forward stepwise.
# ================================================================
def trex_vd_afs_gauss_select(X, y, *, tFDR, K, L_factor, Tmax, rho, seed):
    opt = TRexOptions()
    opt.tFDR = float(tFDR)
    opt.K = int(K)
    opt.L_factor = int(L_factor)
    opt.T_stop = int(Tmax)
    opt.seed = int(seed)
    opt.verbose = False
    opt.solver = SolverType.AFS  # type: ignore
    opt.rho = float(rho)
    opt.calib = CalibMode.CalibrateT
    opt.dummy_law = VDDummyLaw.Spherical
    opt.n_threads = 1
    opt.max_stale_strides = 999
    opt.stride_width = 1
    opt.posthoc_mode = False

    sel = TRexSelector(opt)
    # Gaussian AFS is least-squares based -> feed the centered binary response.
    y_c = np.asarray(y, dtype=float)
    y_c = y_c - y_c.mean()
    res = sel.run(np.asfortranarray(X), np.ascontiguousarray(y_c, dtype=float))
    selected = np.asarray(res.selected_var, dtype=int)
    T_star = int(res.T_stop)
    v_star = float(res.v_thresh)
    return selected, T_star, v_star, len(selected)


# ================================================================
# Model-X knockoff competitor (knockoff+ threshold, see knockoff_glm.py)
# ================================================================
def knockoff_select(X, y, *, alpha, seed):
    out = kog.glm_knockoff(np.ascontiguousarray(X, dtype=float),
                           np.asarray(y, dtype=float), "bernoulli",
                           alpha=float(alpha), seed=int(seed), cfg=KO_CFG)
    return out["selected"]


# ================================================================
# One Monte Carlo replicate
# ================================================================
def one_rep(rep, snr, alpha, K, L, Tmax, rho, seed0, *, n, p, s):
    mix = (rep + 1) * 10_000_019 + int(1_000_003 * snr) + (L + 1) * 97
    rng = np.random.default_rng(int(seed0 + mix))
    X, beta_base, support = make_problem(n=n, p=p, s=s, rng=rng)
    y, snr_exact, prev_eff = sample_y_snr(X, beta_base, float(snr), rng,
                                        prevalence=PREVALENCE)
    sel_ad, *_ = trex_ad_select(
        X, y, tFDR=alpha, K=K, num_dummies=X.shape[1]*L, Tmax=Tmax, rho=rho,
        seed=int(rng.integers(2**31)),
    )
    sel_vd, *_ = trex_vd_select(
        X, y, tFDR=alpha, K=K, L_factor=L, Tmax=Tmax, rho=rho,
        seed=int(rng.integers(2**31)),
    )
    sel_gs, *_ = trex_vd_afs_gauss_select(
        X, y, tFDR=alpha, K=K, L_factor=L, Tmax=Tmax, rho=rho,
        seed=int(rng.integers(2**31)),
    )
    # Drawn last, so the T-Rex arms above keep their seeds. One bad replicate
    # must not take down a long job, so the competitor may fail alone.
    out = {}
    if KNOCKOFF_OK:
        try:
            sel_ko = knockoff_select(
                X, y, alpha=alpha, seed=int(rng.integers(2**31)))
            out["fdp_ko"], out["tpp_ko"] = fdr_tpp(sel_ko, support)
        except Exception as exc:                       # pragma: no cover
            print(f"  ! knockoff arm failed on rep {rep}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    for key, sel in (("ad", sel_ad), ("vd", sel_vd), ("gs", sel_gs)):
        out[f"fdp_{key}"], out[f"tpp_{key}"] = fdr_tpp(sel, support)
    out.update({"L": int(L), "snr": float(snr), "rep": int(rep),
                "snr_exact": snr_exact, "prev_eff": prev_eff})
    return out


METHODS = {
    "ad": "AD-AFS-Log-T-Rex",
    "vd": "VD-AFS-Log-T-Rex",
    "gs": "VD-AFS-Gauss",
    "ko": "MX-Knockoff-Log",
}


# ================================================================
# Sweep: means and Monte Carlo standard errors per method, SNR and L
# ================================================================
def sweep(snrs, Ls, *, alpha=0.1, reps=50, K=20, Tmax=20, rho=1.0,
          seed0=0, n_jobs=None, n=300, p=1000, s=10):
    if n_jobs is None:
        n_jobs = slurm_n_jobs()
    print(f"[sweep] n_jobs={n_jobs}  est. AD-arm peak RSS = "
          f"{ad_arm_mem_estimate_gb(n, p, max(Ls), K, n_jobs):.1f} GB", flush=True)
    if not KNOCKOFF_OK:
        print(f"! knockoff arm skipped: {KNOCKOFF_WHY}", flush=True)
    rows = []
    with tqdm_joblib(tqdm(total=len(Ls) * len(snrs) * reps, desc="MC tasks")):
        for L in Ls:
            for snr in snrs:
                res = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(one_rep)(
                        r, float(snr), float(alpha), int(K), int(L), int(Tmax),
                        float(rho), int(seed0), n=n, p=p, s=s,
                    ) for r in range(reps)
                )
                res = [r for r in res if r is not None]
                if not res:
                    continue
                extra = {c: float(np.mean([d[c] for d in res])) for c in ("snr_exact", "prev_eff")}
                for key, name in METHODS.items():
                    vals = [d for d in res if f"fdp_{key}" in d]
                    if not vals:
                        continue
                    f = np.array([d[f"fdp_{key}"] for d in vals])
                    t = np.array([d[f"tpp_{key}"] for d in vals])
                    r_ = np.sqrt(len(vals))
                    rows.append({
                        "L": L, "snr": snr, **extra, "method": name,
                        "fdp": float(f.mean()), "tpp": float(t.mean()),
                        "fdp_se": float(f.std(ddof=1) / r_) if len(vals) > 1 else np.nan,
                        "tpp_se": float(t.std(ddof=1) / r_) if len(vals) > 1 else np.nan,
                    })
    return pd.DataFrame(rows)


# ================================================================
# Main: writes results/sec4_1_equivalence/logistic/summary.csv and meta.json
# ================================================================
if __name__ == "__main__":
    outdir = Path("results") / "sec4_1_equivalence" / "logistic"
    outdir.mkdir(parents=True, exist_ok=True)

    snrs = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    Ls = (5,)
    alpha = 0.1
    reps = 500
    K = 20
    Tmax = 30
    rho = 1
    seed0 = 123
    n, p, s = 300, 1000, 10
    max_workers = 50      # the AD arm holds K augmented designs per worker

    df = sweep(snrs=snrs, Ls=Ls, alpha=alpha, reps=reps, K=K,
               Tmax=Tmax, rho=rho, seed0=seed0, n_jobs=slurm_n_jobs(cap=max_workers),
               n=n, p=p, s=s)
    df.to_csv(outdir / "summary.csv", index=False)
    meta = dict(snrs=list(snrs), prevalence=PREVALENCE, Ls=list(Ls), alpha=alpha, reps=reps,
                K=K, Tmax=Tmax, rho=rho, seed0=seed0, n=n, p=p, s=s)
    json.dump(meta, open(outdir / "meta.json", "w"), indent=2)
    print(f"Saved to {outdir}")