// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// vd_afs_poisson.hpp — Virtual Dummy AFS with Poisson (log) link function.
//
// score_direction = y_count - exp(b0 + X_A * beta)
// Basis grows by orthogonalizing the score (not the predictor).
// IRLS refit replaces OLS. An always-on intercept b0 is fitted jointly.
// At rho=1: reduces to GLM-OMP (full IRLS refit each step).
// At rho<1: damped blend of old coefficients and IRLS solution.
#pragma once
#include "vd_base.hpp"

class VD_AFS_Poisson : public VD_Base {
public:
  using VD_Base::VD_Base;

  MatC run(int T = 1) override;

private:
  // ---- Poisson-specific state ----
  double rho_ = 1.0;
  Vec y_count_;      // non-negative counts (raw y)
  Vec eta_;          // linear predictor: b0 + X_A * beta (active part)
  Vec mu_pois_;      // exp(eta)
  Vec score_;        // y_count - mu
  Vec nu_active_;    // IRLS solution on active set
  bool irls_stale_ = true;

  // Intercept (log-link baseline). Without it mu is stuck at exp(0)=1 while
  // E[y]=base_rate, so the score y-mu carries the mean level and leaks it into
  // the uncentered dummies, inflating the FDR threshold. The intercept enforces
  // sum_i (y_i - mu_i) = 0, i.e. a mean-zero score, killing that leak.
  double b0_      = 0.0;  // blended intercept, part of eta_
  double b0_irls_ = 0.0;  // latest IRLS intercept solution

  // Clamp on eta to keep exp() from overflowing.
  static constexpr double kEtaClamp = 30.0;

  // ---- Score override ----
  const Vec& score_direction_() const override { return score_; }

  // ---- Poisson helpers ----
  static Vec mean_fn_(const Vec& eta);   // exp(clamp(eta))
  void update_score_();                  // recompute mu, score from eta

  // ---- Selection ----
  struct Candidate {
    enum class Pool : uint8_t { Real, VD, RealizedDummy };
    Pool pool; int index; double abs_corr; bool is_new;
  };
  std::optional<Candidate> find_best_candidate_() const;

  // ---- IRLS ----
  void irls_solve_(int max_iter = 25, double tol = 1e-8);

  // ---- Blend ----
  void afs_blend_();

  // ---- Init ----
  void init_poisson_();
  bool poisson_inited_ = false;
};
