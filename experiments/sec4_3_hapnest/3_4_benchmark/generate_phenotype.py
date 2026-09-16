#!/usr/bin/env python
"""
Disease-subtype phenotype of Sec. 4.3 on HAPNEST genotypes.

Class 0 is the healthy reference, classes 1..C-1 are disease subtypes. The
per-subtype log relative risk of individual i is

    eta[i, c] = sum_{j in causal} g[i, j] * log rr[j, c],   c = 1..C-1,

with g in {0, 1, 2} the dosage, so a heterozygote carries rr and a homozygote
rr^2 (multiplicative relative risk). Of the s causal SNPs, n_shared act on all
subtypes with rr ~ Unif(het_rr_shared) and a per-class jitter of +-15%; the
others act on one subtype with rr ~ Unif(het_rr_specific). The sign of each
log relative risk is drawn at random. Labels are drawn from the softmax, with
intercepts calibrated so the expected class shares equal class_prev.

    python generate_phenotype.py <data_dir> <pheno_dir> <run_id> [cfg_json]

Writes <pheno_dir>/pheno_run_<run_id>.npz with the raw integer labels y (not
centered), the causal indices and which of them are shared.
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), adapted from the binary HAPNEST pipeline of Koka et al. (2026). GPLv3, see NOTICE.

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

warnings.filterwarnings("ignore")


# =============================================================
# Causal architecture -> log relative risk matrix
# =============================================================

def build_log_rr(s, C, n_shared, het_rr_shared, het_rr_specific, rng):
    """Return (log_rr, is_shared, causal_class).

    log_rr has shape (s, C-1); row k holds the log het. relative risks of the
    k-th causal SNP for subtypes 1..C-1. A homozygote gets 2 * log_rr, i.e.
    rr^2, matching the binary multiplicative-RR generator.
    """
    K = C - 1
    log_rr = np.zeros((s, K))
    is_shared = np.zeros(s, dtype=bool)
    causal_class = np.full(s, -1, dtype=int)

    n_shared = int(min(n_shared, s))
    shared_pos = rng.choice(s, size=n_shared, replace=False)
    is_shared[shared_pos] = True
    for k in range(s):
        if is_shared[k]:
            rr = rng.uniform(*het_rr_shared)
            sign = rng.choice([-1.0, 1.0])
            log_rr[k, :] = sign * np.log(rr) * rng.uniform(0.85, 1.15, size=K)
        else:
            c = int(rng.integers(0, K))
            rr = rng.uniform(*het_rr_specific)
            log_rr[k, c] = np.log(rr) * rng.choice([-1.0, 1.0])
            causal_class[k] = c

    return log_rr, is_shared, causal_class


def linear_predictor(X_raw, causal_idx, log_rr, rng):
    """eta of shape (n, C-1) from raw 0/1/2 dosages."""
    n = X_raw.shape[0]
    K = log_rr.shape[1]
    eta = np.zeros((n, K))
    # Which allele is the risk allele is arbitrary in HAPNEST's coding, so flip
    # at random exactly as the binary generator does.
    risk_allele = rng.integers(0, 2, size=len(causal_idx))
    for k, (j, ra) in enumerate(zip(causal_idx, risk_allele)):
        g = np.round(np.asarray(X_raw[:, j], dtype=float)).astype(int).clip(0, 2)
        if ra == 0:
            g = 2 - g
        eta += np.outer(g.astype(float), log_rr[k])
    return eta


# =============================================================
# Label sampling
# =============================================================

def _softmax_probs(eta, b):
    """Reference-category softmax. eta (n, K), b (K,) -> Pr (n, C)."""
    z = np.clip(eta + b[None, :], -30.0, 30.0)
    E = np.concatenate([np.ones((eta.shape[0], 1)), np.exp(z)], axis=1)
    return E / E.sum(axis=1, keepdims=True)


def calibrate_intercepts(eta, class_prev, tol=1e-8, max_iter=200):
    """Newton solve for b so that mean_i Pr(y_i = c) == class_prev[c].

    The map b -> mean class probabilities is the gradient of a convex function,
    so plain Newton on the (K x K) Jacobian converges in a handful of steps.
    """
    n, K = eta.shape
    target = np.asarray(class_prev, dtype=float)[1:]          # classes 1..K
    b = np.log(target / class_prev[0])                        # marginal start
    for _ in range(max_iter):
        Pr = _softmax_probs(eta, b)
        P = Pr[:, 1:]                                         # (n, K)
        g = P.mean(axis=0) - target
        if np.max(np.abs(g)) < tol:
            break
        # d mean_c / d b_c' = mean_i [ P_ic (delta_cc' - P_ic') ]
        J = (np.diag(P.mean(axis=0)) - (P.T @ P) / n)
        step = np.linalg.solve(J + 1e-12 * np.eye(K), g)
        b -= step
    return b, _softmax_probs(eta, b)


def sample_softmax(eta, class_prev, rng):
    b, Pr = calibrate_intercepts(eta, class_prev)
    C = Pr.shape[1]
    # Vectorised categorical draw: one uniform per row against the cdf.
    u = rng.random(Pr.shape[0])[:, None]
    y = (u > np.cumsum(Pr, axis=1)).sum(axis=1).astype(float)
    y = np.clip(y, 0, C - 1)
    diag = {
        "intercepts": b.tolist(),
        "bayes_acc": float(Pr.max(axis=1).mean()),
        "realized_shares": (np.bincount(y.astype(int), minlength=C) / len(y)).tolist(),
    }
    return y, diag


# =============================================================
# Data loading (identical contract to generate_phenotype.py)
# =============================================================

def load_data(data_dir, run_id):
    run_dir = Path(data_dir) / f"run_{run_id}"
    meta_path = run_dir / "meta.json"
    xstd_dat = run_dir / "X_std.dat"
    xraw_dat = run_dir / "X_raw.dat"
    npz_path = run_dir / "data.npz"

    if xstd_dat.exists() and xraw_dat.exists() and meta_path.exists():
        meta = json.load(open(meta_path))
        n, p = meta["n_samples"], meta["p_pruned_total"]
        Xc = np.memmap(xstd_dat, dtype="float64", mode="r", shape=(n, p), order="F")
        X_raw = np.memmap(xraw_dat, dtype="float64", mode="r", shape=(n, p), order="F")
    elif npz_path.exists():
        d = np.load(npz_path)
        Xc = np.asarray(d["X"], dtype=np.float64)
        X_raw = np.asarray(d["X_raw"], dtype=np.float64)
        d.close()
        n, p = Xc.shape
    else:
        return None

    return Xc, X_raw, n, p


# =============================================================
# Entry point
# =============================================================

# The setting of the paper: C = 4, half controls, 12 causal SNPs, 4 shared.
DEFAULTS = dict(
    C=4,
    s=12,
    n_shared=4,
    class_prev=[0.5, 1 / 6, 1 / 6, 1 / 6],
    het_rr_shared=[1.15, 1.30],
    het_rr_specific=[1.20, 1.42],
    seed0=42,
)


def generate(X_raw, p, cfg, rng):
    C = int(cfg["C"])
    s = int(cfg["s"])
    class_prev = np.asarray(cfg["class_prev"], dtype=float)
    if len(class_prev) != C:
        raise ValueError(f"class_prev has {len(class_prev)} entries, need C={C}")
    class_prev = class_prev / class_prev.sum()

    causal_idx = np.sort(rng.choice(p, size=s, replace=False)).astype(int)
    log_rr, is_shared, causal_class = build_log_rr(
        s, C, cfg["n_shared"],
        tuple(cfg["het_rr_shared"]), tuple(cfg["het_rr_specific"]), rng,
    )
    eta = linear_predictor(X_raw, causal_idx, log_rr, rng)

    y, diag = sample_softmax(eta, class_prev, rng)

    return y, causal_idx, log_rr, is_shared, causal_class, diag


def main():
    data_dir = sys.argv[1]
    pheno_dir = Path(sys.argv[2])
    run_id = int(sys.argv[3])
    cfg = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}

    for k, v in DEFAULTS.items():
        cfg.setdefault(k, v)

    pheno_dir.mkdir(parents=True, exist_ok=True)
    pheno_path = pheno_dir / f"pheno_run_{run_id}.npz"
    if pheno_path.exists():
        print(f"run {run_id}: already exists, skipping")
        return

    loaded = load_data(data_dir, run_id)
    if loaded is None:
        print(f"run {run_id}: data not found")
        sys.exit(1)
    Xc, X_raw, n, p = loaded

    seed = int(cfg["seed0"]) + run_id * 10_007
    rng = np.random.default_rng(seed)

    y, causal_idx, log_rr, is_shared, causal_class, diag = generate(
        X_raw, p, cfg, rng)

    C = int(cfg["C"])
    counts = np.bincount(y.astype(int), minlength=C)

    save = dict(
        y=y,                                   # raw labels, NOT centered
        causal_idx=causal_idx,
        is_shared=is_shared,
        causal_class=causal_class,
        log_rr=log_rr,
        class_counts=counts,
        C=C, n=n, p=p, seed=seed,
    )

    np.savez_compressed(pheno_path, **save)

    print(f"run {run_id}: n={n} p={p} C={C} s={len(causal_idx)} "
          f"({int(is_shared.sum())} shared) counts={counts.tolist()} "
          f"bayes_acc={diag['bayes_acc']}")


if __name__ == "__main__":
    main()
