// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// vd_afs_multinomial.hpp — Virtual Dummy AFS with multinomial (softmax) link.
//
// Multi-class classification, reference-category parametrization (class 0 is the
// baseline). For C classes there are Cm = C-1 non-reference score directions
//
//     s^(c) = y^(c) - p^(c),   c = 1..C-1,   p = softmax(eta),
//
// each an n-vector in the centered subspace H. This is the natural
// generalization of the binary logistic selector: the single residual becomes
// Cm residuals, and a candidate is scored by the L2 norm across classes of its
// per-class inner products with the score directions.
//
// The norm is taken over ALL C classes, not over the Cm tracked ones. The full
// C-class score vector sums to zero, so the held-out entry is recoverable from
// the tracked ones and the resulting rank statistic is invariant to the choice
// of reference class and to any relabeling of the classes (see
// class_score_norm in the .cpp). Ranking on the truncated Cm-vector would make
// the selected set depend on an arbitrary labeling convention.
//
// VD-compatibility (Koka et al. 2026, Appendix A / Table 2): the selection rule
// depends on the dummies ONLY through the inner products <d, s^(c)>, and all
// Cm score directions are appended to the revealed subspace V_k. Therefore the
// basis grows by up to Cm axes per step (the paper explicitly permits "one or
// more score directions"). Crucially, we use the UNWEIGHTED per-class score,
// NOT the Fisher-information-standardized multinomial score test: the latter
// would require sum_i v_i^2 (W_i)_cc', a functional of the squared coordinates
// that is NOT recoverable from the tracked V_k-projections, which would break
// compatibility (Table 2, "curvature information").
//
// At rho=1: full multinomial IRLS refit each step. At rho<1: damped AFS blend.
#pragma once
#include "vd_base.hpp"

class VD_AFS_Multinomial : public VD_Base {
public:
  using VD_Base::VD_Base;

  MatC run(int T = 1) override;

private:
  // ---- Problem geometry ----
  int C_  = 0;   // number of classes (labels 0..C_-1)
  int Cm_ = 0;   // C_ - 1 non-reference classes

  // ---- Multinomial state (all n x Cm_ unless noted) ----
  double rho_ = 1.0;
  MatC Y_ind_;      // {0,1} indicator columns for classes 1..C-1
  MatC eta_;        // linear predictor b0_c + X_A * beta_c
  MatC prob_;       // softmax(eta) for the non-reference classes
  MatC score_;      // Y_ind - prob, each column centered to H

  Vec  b0_;         // (Cm_) blended intercepts, part of eta_
  Vec  b0_irls_;    // (Cm_) latest IRLS intercepts

  // Per-class coefficients. beta_multi_ holds real-feature coeffs (p x Cm_),
  // beta_dummy_multi_ the realized-dummy coeffs (T_stop x Cm_). The base-class
  // beta_ carries only a scalar per feature for the returned path; we fill it
  // with the across-class L2 norm so beta_real() stays interpretable.
  MatC beta_multi_;
  MatC beta_dummy_multi_;
  MatC nu_active_;  // (k x Cm_) IRLS solution on the active set (excl. intercept)

  bool irls_stale_ = true;

  // Clamp on eta to keep exp() from overflowing in the softmax.
  static constexpr double kEtaClamp = 30.0;

  // ---- Helpers ----
  void softmax_(const MatC& eta, MatC& prob) const;  // reference-category softmax
  void update_score_();                              // recompute prob, centered score

  // Multi-direction analogues of the single-score base-class hooks. We do NOT
  // use grow_basis_from_score_() / full_corr_refresh_() from the base, since
  // those assume a single score vector.
  void grow_basis_from_scores_();   // append every s^(c) not yet in V_k
  void full_corr_refresh_multi_();  // per-class corr aggregated to an L2 norm

  // ---- Selection ----
  struct Candidate {
    enum class Pool : uint8_t { Real, VD, RealizedDummy };
    Pool pool; int index; double abs_corr; bool is_new;
  };
  std::optional<Candidate> find_best_candidate_() const;

  // ---- IRLS + blend ----
  void irls_solve_(int max_iter = 25, double tol = 1e-8);
  void afs_blend_();

  // ---- Init ----
  void init_multinomial_();
  bool multinomial_inited_ = false;
};
