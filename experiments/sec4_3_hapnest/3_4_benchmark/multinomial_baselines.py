#!/usr/bin/env python
"""
Marginal competitor of Sec. 4.3: the multinomial score test, a C-class
generalisation of the Cochran-Armitage trend test, followed by BH or BY.

Under the intercept-only null the fitted class probabilities are the observed
frequencies pihat (class 0 is the reference). With Y the (n x C-1) indicator of
classes 1..C-1 and g_j the centered dosage of SNP j,

    U_j = g_j^T (Y - 1 pihat^T),   Sigma = diag(pihat) - pihat pihat^T,
    T_j = U_j^T Sigma^{-1} U_j / ||g_j||^2  ~  chi^2_{C-1}  under H0.

Sigma is inverted once for all SNPs.
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), adapted from the binary HAPNEST pipeline of Koka et al. (2026). GPLv3, see NOTICE.

from __future__ import annotations


import numpy as np
from scipy import stats


EPS = 1e-12


# =============================================================
# Generalised Cochran-Armitage (multinomial score test)
# =============================================================

def multinomial_score_pvalues(X_raw, y, C=None, block=8192):
    """Per-SNP chi^2_{C-1} p-values. X_raw may be a memmap; processed in blocks.

    Returns (pvals, info) with pvals of shape (p,).
    """
    y = np.asarray(y, dtype=int).ravel()
    n = y.shape[0]
    p = X_raw.shape[1]
    if C is None:
        C = int(y.max()) + 1
    K = C - 1

    counts = np.bincount(y, minlength=C).astype(float)
    pihat = counts[1:] / n                                   # classes 1..K
    Sigma = np.diag(pihat) - np.outer(pihat, pihat)
    Sigma_inv = np.linalg.pinv(Sigma)

    # Residual one-hot matrix, centered so the intercept score vanishes.
    Yc = np.zeros((n, K))
    for c in range(1, C):
        Yc[:, c - 1] = (y == c).astype(float)
    Yc -= pihat[None, :]

    T = np.empty(p)
    for start in range(0, p, block):
        stop = min(start + block, p)
        G = np.asarray(X_raw[:, start:stop], dtype=np.float64)
        G = G - G.mean(axis=0, keepdims=True)
        ss = np.einsum("ij,ij->j", G, G)                     # ||gtilde_j||^2
        U = G.T @ Yc                                         # (b, K)
        quad = np.einsum("bk,kl,bl->b", U, Sigma_inv, U)
        T[start:stop] = np.where(ss > EPS, quad / np.maximum(ss, EPS), 0.0)

    pvals = stats.chi2.sf(T, df=K)
    return pvals, {"test_mode": "multinomial_score", "df": K, "C": C}


# =============================================================
# Classical Cochran-Armitage (C = 2), used by the two references
# =============================================================


# =============================================================
# BH / BY wrappers
# =============================================================

def bh_by_select(pvals, alpha, method="fdr_bh"):
    from statsmodels.stats.multitest import multipletests
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method=method)
    return np.where(reject)[0].astype(int)





