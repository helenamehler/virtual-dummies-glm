// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// vd_afs_cox.hpp — Virtual Dummy AFS with Cox proportional-hazards link.
//
// Survival / time-to-event selection. The response y carries the non-negative
// event/censoring TIMES; the censoring indicator delta (1 = event observed,
// 0 = right-censored) is supplied separately in VDOptions::cox_status.
//
// The Cox partial log-likelihood (Breslow ties) is
//
//     l(beta) = sum_{i: delta_i=1} [ eta_i - log( sum_{j in R(t_i)} exp(eta_j) ) ],
//
// with eta = X_A beta and risk set R(t) = { j : t_j >= t }. There is NO
// intercept: any constant added to eta cancels inside the risk-set ratio, so
// the baseline hazard already plays the intercept's role.
//
// score_direction = martingale residual  s_i = delta_i - exp(eta_i) * H0_hat(t_i),
// where H0_hat is the Breslow baseline cumulative hazard. Under Breslow ties the
// martingale residuals sum to EXACTLY zero, so the score lives in the centered
// subspace H by construction (no projection needed, unlike logistic/Poisson).
//
// Basis grows by orthogonalizing the score (not the predictor). A Cox
// Newton-Raphson refit on the active set replaces IRLS/OLS.
// At rho=1: full Newton refit each step (GLM-OMP analogue). At rho<1: damped
// AFS blend of old coefficients and the Newton solution.
#pragma once
#include "vd_base.hpp"

class VD_AFS_Cox : public VD_Base {
public:
  using VD_Base::VD_Base;

  MatC run(int T = 1) override;

private:
  // ---- Cox-specific state ----
  double rho_ = 1.0;
  Vec    time_;        // event/censoring times (alias of y_, kept explicit)
  Vec    delta_;       // {0,1} censoring indicator (1 = event)
  Vec    eta_;         // linear predictor X_A * beta (NO intercept)
  Vec    w_;           // exp(eta) (risk weights)
  Vec    score_;       // martingale residuals delta - w * H0_hat
  Vec    nu_active_;   // Newton solution on the active set
  bool   newton_stale_ = true;
  bool   efron_ = false;

  // Precomputed time ordering (times never change, so sort once).
  std::vector<int> ord_desc_;   // sample indices sorted by DESCENDING time
  int              n_events_ = 0;

  // Clamp on eta to keep exp() finite.
  static constexpr double kEtaClamp = 30.0;

  // ---- Score override ----
  const Vec& score_direction_() const override { return score_; }

  // ---- Cox helpers ----
  // Martingale-residual score for a given linear predictor eta (Breslow).
  void compute_score_(const Vec& eta, Vec& score_out) const;
  // Refresh w_, score_ from the current eta_.
  void update_score_();
  // Cox Newton-Raphson gradient g (k) and Hessian H (k x k) on design A at eta.
  void cox_grad_hess_(const MatC& A, const Vec& eta,
                      Vec& g, MatC& H) const;

  // ---- Selection (identical rule to the other AFS GLMs) ----
  struct Candidate {
    enum class Pool : uint8_t { Real, VD, RealizedDummy };
    Pool pool; int index; double abs_corr; bool is_new;
  };
  std::optional<Candidate> find_best_candidate_() const;

  // ---- Newton refit + blend ----
  void newton_solve_(int max_iter = 25, double tol = 1e-8);
  void afs_blend_();

  // ---- Init ----
  void init_cox_();
  bool cox_inited_ = false;
};
