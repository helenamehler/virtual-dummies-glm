"""
SNR definitions of Sec. 4.1 and the calibration that hits a target SNR.

Response-scale SNR, from the law of total variance:

    SNR = Var{E(y|x)} / E{Var(y|x)}

    Gaussian     Var(X beta) / sigma^2
    Bernoulli    Var(mu) / E[mu (1 - mu)]
    Poisson      Var(lambda) / E[lambda]
    Multinomial  sum_c Var(pi_c) / (1 - E sum_c pi_c^2)

Censored event times have no conditional variance function, so for Cox the SNR
is taken on the latent log-time scale. With an exponential baseline hazard,
log T = -eta + xi with a Gumbel error of variance pi^2/6, so

    SNR = Var(eta) / (pi^2/6).

The linear predictor is standardised and scaled, eta = c * eta_unit. For
Gaussian and Cox c has a closed form, otherwise it is found by bisection on the
quantity above.
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
from __future__ import annotations
import numpy as np

EPS = 1e-12


# ======================================================================
# Die SNR-Definitionen
# ======================================================================
def snr_gaussian(eta, sigma=1.0):
    """Var(X beta) / sigma^2."""
    return float(np.asarray(eta, float).var()) / float(sigma) ** 2


def snr_bernoulli(mu):
    """Var(mu) / E[mu (1-mu)]."""
    mu = np.asarray(mu, float)
    return float(mu.var() / max(np.mean(mu * (1.0 - mu)), EPS))


def snr_poisson(lam):
    """Var(lam) / E[lam] -- der Dispersionsindex."""
    lam = np.asarray(lam, float)
    return float(lam.var() / max(lam.mean(), EPS))


def snr_multinomial(Pr):
    """sum_c Var(pi_c) / (1 - E sum_c pi_c^2).

    Pr : (n, C) Klassenwahrscheinlichkeiten. Spur-Version der
    Varianzzerlegung; reduziert sich fuer C = 2 auf snr_bernoulli.
    """
    Pr = np.asarray(Pr, float)
    return float(Pr.var(axis=0).sum()) / max(
        float(np.mean(1.0 - (Pr ** 2).sum(axis=1))), EPS)


GUMBEL_VAR = np.pi ** 2 / 6.0      # Var(log E), E ~ Exp(1)


def snr_cox(eta):
    """Var(eta) / (pi^2/6) -- SNR auf der latenten Log-Zeit-Skala.

    Bei exponentieller Baseline ist T | x ~ Exp(exp(eta)), also
    log T = -eta + log E mit E ~ Exp(1), und log E ist standard-Gumbel mit
    Varianz pi^2/6. Die Zensierung geht bewusst nicht ein, siehe den
    Modul-Docstring.
    """
    return float(np.asarray(eta, float).var()) / GUMBEL_VAR


# ======================================================================
# Kalibrierung
# ======================================================================
def unit_variance(v, axis=0):
    """Normiert (spaltenweise) auf Einheitsvarianz."""
    v = np.asarray(v, float)
    return v / np.sqrt(np.maximum(v.var(axis=axis, keepdims=True), EPS))


def scale_gaussian(snr_target, sigma=1.0):
    """Geschlossen: SNR = c^2 / sigma^2  =>  c = sigma sqrt(SNR)."""
    return float(sigma) * float(np.sqrt(max(float(snr_target), 0.0)))


def scale_cox(snr_target):
    """Geschlossen: SNR = a^2 / (pi^2/6)  =>  a = sqrt(SNR * pi^2/6)."""
    return float(np.sqrt(max(float(snr_target), 0.0) * GUMBEL_VAR))


def calibrate(snr_target, snr_of_scale, lo=0.0, hi=None, iters=60, tol=1e-10):
    """Skalenfaktor c mit snr_of_scale(c) == snr_target (Bisektion).

    ``snr_of_scale`` bildet den Skalenfaktor auf die realisierte SNR ab und
    muss monoton wachsend sein. Fuer Bernoulli, Poisson und Multinomial mit
    varianznormiertem Praediktor ist das ueber den gesamten hier verwendeten
    Bereich der Fall.
    """
    t = float(snr_target)
    if t <= 0.0:
        return 0.0
    if hi is None:
        hi = 1.0
        while snr_of_scale(hi) < t and hi < 1e4:
            hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if snr_of_scale(mid) < t:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)
