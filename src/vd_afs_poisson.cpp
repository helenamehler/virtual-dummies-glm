// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// vd_afs_poisson.cpp -- GLM forward stepwise with Poisson (log) link.
#include "vd_afs_poisson.hpp"

#include <algorithm>
#include <cmath>

// ---------- Mean function: mu = exp(eta), with eta clamped ----------
Vec VD_AFS_Poisson::mean_fn_(const Vec& eta) {
  return eta.array().min(kEtaClamp).max(-kEtaClamp).exp().matrix();
}

// ---------- Update score from current eta ----------
void VD_AFS_Poisson::update_score_() {
  mu_pois_ = mean_fn_(eta_);
  score_ = y_count_ - mu_pois_;

  // The real design and virtual dummies live in the centered subspace H.
  // With an intercept and AFS blending, the raw score is not guaranteed to be
  // exactly mean-zero at every intermediate step. Project it back to H before
  // using it for VD basis growth or correlations.
  score_.array() -= score_.mean();

  // Keep base-class aliases consistent with the GLM score.
  mu_ = mu_pois_;
  residuals_ = score_;
}

// ---------- Init ----------
void VD_AFS_Poisson::init_poisson_() {
  if (poisson_inited_) return;

  rho_ = std::max(1e-6, std::min(opt_.rho, 1.0));

  // Poisson responses are raw non-negative counts.
  y_count_.resize(n_);
  for (int i = 0; i < n_; ++i)
    y_count_(i) = std::max(0.0, y_(i));

  // Intercept-only null model.
  // Null MLE for Poisson with log link: exp(b0) = mean(y).
  const double ybar = std::max(y_count_.mean(), 1e-8);
  b0_ = std::log(ybar);
  b0_irls_ = b0_;
  eta_ = Vec::Constant(n_, b0_);

  // Initializes mu_pois_, score_, mu_, and residuals_. The score is centered.
  update_score_();

  // VD_Base initialized the first VD direction from raw y_. For Poisson GLMs
  // the selector uses the score y - mu, so the initial VD basis/projections
  // must be rebuilt from the centered null score instead.
  basis_indices_.clear();
  vd_rows_filled_ = 0;
  vd_stick_ = Vec::Ones(L_);
  vd_corr_ = Vec::Zero(L_);
  std::fill(vd_is_realized_.begin(), vd_is_realized_.end(), 0);
  vd_unrealized_idx_.clear();
  T_realized_ = 0;

  const double sn = score_.norm();
  if (sn > opt_.eps * std::sqrt(double(n_))) {
    y_norm_ = sn;
    basis_.col(0) = score_ / sn;
    basis_indices_.push_back(VD_Y_SENTINEL);
    initialize_virtual_dummies_();
  }

  // Compute X^T score, VD correlations, and realized-dummy correlations using
  // the Poisson score rather than raw y_.
  full_corr_refresh_();

  nu_active_.resize(0);
  irls_stale_ = true;
  poisson_inited_ = true;
}

// ---------- Candidate search (same as AFS: scan ALL features) ----------
std::optional<VD_AFS_Poisson::Candidate>
VD_AFS_Poisson::find_best_candidate_() const {
  Candidate best{Candidate::Pool::Real, -1, 0.0, true};

  // 1. Check virtual dummies first. With strict '>' below, exact ties stay with
  // the dummy/null competitor, which is conservative.
  for (int d = 0; d < L_; ++d) {
    if (vd_is_realized_[d]) continue;
    const double ac = std::abs(vd_corr_(d));
    if (ac > best.abs_corr) best = {Candidate::Pool::VD, d, ac, true};
  }

  // 2. Check realized dummies, which are also null competitors. They can be
  // selected again by AFS, but they are not new active columns.
  for (int j = 0; j < T_realized_; ++j) {
    const double ac = std::abs(corr_realized_(j));
    if (ac > best.abs_corr)
      best = {Candidate::Pool::RealizedDummy, j, ac, false};
  }

  // 3. Check real features last. If an already active real feature wins, AFS
  // performs another shrinkage/blending step without changing the active set.
  for (int j = 0; j < p_; ++j) {
    const double ac = std::abs(corr_(j));
    if (ac > best.abs_corr)
      best = {Candidate::Pool::Real, j, ac, !is_active_[j]};
  }

  if (best.index < 0 || best.abs_corr < 100.0 * opt_.eps)
    return std::nullopt;
  return best;
}

// ---------- IRLS solve on active set (intercept + active features) ----------
void VD_AFS_Poisson::irls_solve_(int max_iter, double tol) {
  const int k = static_cast<int>(active_features_.size());
  const int kk = k + 1;  // +1 for the always-on intercept column

  // Parameter vector theta = [b0; nu_active(0..k-1)]. Start from the current
  // blended values for a warm start.
  Vec theta(kk);
  theta(0) = b0_;
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Real) {
      theta(i + 1) = beta_(af.index);
    } else if (af.index >= 0 && af.index < beta_dummy_.size()) {
      theta(i + 1) = beta_dummy_(af.index);
    } else {
      theta(i + 1) = 0.0;
    }
  }

  // Augmented design [1 | X_active]. Columns are fixed during IRLS.
  MatC Xa(n_, kk);
  Xa.col(0).setOnes();
  for (int j = 0; j < k; ++j)
    Xa.col(j + 1) = X_active_.col(j);

  const double w_lo = 1e-10;  // floor on Poisson weights (= mu)

  for (int iter = 0; iter < max_iter; ++iter) {
    // Linear predictor including intercept.
    const Vec eta_a = Xa * theta;

    // Poisson mean, clamped via mean_fn_().
    const Vec mu_a = mean_fn_(eta_a);

    // Weights (= mu) and working response.
    const Vec w = mu_a.cwiseMax(w_lo);
    const Vec z = eta_a.array() +
                  (y_count_.array() - mu_a.array()) / w.array();

    // Weighted normal equations: (Xa^T W Xa) theta = Xa^T W z.
    const Vec sqrt_w = w.cwiseSqrt();
    MatC Xw(n_, kk);
    for (int j = 0; j < kk; ++j)
      Xw.col(j) = Xa.col(j).cwiseProduct(sqrt_w);
    const Vec zw = z.cwiseProduct(sqrt_w);

    MatC G = Xw.transpose() * Xw;
    Vec rhs = Xw.transpose() * zw;

    // Small ridge for numerical stability.
    G.diagonal().array() += opt_.eps;

    Eigen::LLT<MatC> llt(G);
    if (llt.info() != Eigen::Success) break;  // fallback: keep current theta

    const Vec theta_new = llt.solve(rhs);
    const double delta = (theta_new - theta).squaredNorm();
    theta = theta_new;

    if (delta < tol * tol * (1.0 + theta.squaredNorm())) break;
  }

  // Split result back into intercept + active-feature coefficients.
  b0_irls_ = theta(0);
  nu_active_.resize(k);
  for (int i = 0; i < k; ++i)
    nu_active_(i) = theta(i + 1);

  irls_stale_ = false;
}

// ---------- AFS blend ----------
void VD_AFS_Poisson::afs_blend_() {
  const int k = static_cast<int>(active_features_.size());
  if (k == 0) return;

  if (irls_stale_) irls_solve_();

  // Blend intercept: b0 <- (1-rho) b0_old + rho b0_irls.
  b0_ = (1.0 - rho_) * b0_ + rho_ * b0_irls_;

  // Blend real coefficients: beta <- (1-rho) beta_old + rho nu.
  beta_ *= (1.0 - rho_);
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Real)
      beta_(af.index) += rho_ * nu_active_(i);
  }

  // Blend realized-dummy coefficients. Do not touch dummy slots that have not
  // been realized yet.
  const int n_dummy_coef = std::min(T_realized_, static_cast<int>(beta_dummy_.size()));
  if (n_dummy_coef > 0)
    beta_dummy_.head(n_dummy_coef) *= (1.0 - rho_);

  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Dummy &&
        af.index >= 0 && af.index < beta_dummy_.size()) {
      beta_dummy_(af.index) += rho_ * nu_active_(i);
    }
  }

  // Recompute linear predictor from blended beta + intercept.
  Vec beta_active(k);
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Real) {
      beta_active(i) = beta_(af.index);
    } else if (af.index >= 0 && af.index < beta_dummy_.size()) {
      beta_active(i) = beta_dummy_(af.index);
    } else {
      beta_active(i) = 0.0;
    }
  }

  eta_ = X_active_.leftCols(k) * beta_active;
  eta_.array() += b0_;

  // Updates mu_pois_, centered score_, mu_, and residuals_. Correlations are
  // refreshed in run(), after the new score direction has been revealed to the
  // VD basis.
  update_score_();
}

// ---------- Run ----------
MatC VD_AFS_Poisson::run(int T) {
  init_poisson_();

  std::vector<Vec> path;
  auto record = [&]() {
    // beta_ holds only real-feature coefficients. The intercept b0_ is fitted
    // but intentionally not part of the selection path.
    Vec s = beta_;
    if (normx_.size() == beta_.size())
      s.array() /= normx_.array();
    path.emplace_back(std::move(s));
  };

  if (step_ == 0) record();

  const int max_steps = p_ + opt_.T_stop;

  for (int it = step_; it < max_steps; ++it) {
    const int prev_T_realized = T_realized_;

    const auto cand = find_best_candidate_();
    if (!cand) break;

    if (cand->is_new) {
      Vec x_col;

      if (cand->pool == Candidate::Pool::VD) {
        realize_dummy_(cand->index, /*append_to_basis=*/false);
        const int jslot = T_realized_ - 1;
        active_features_.push_back({ActiveFeature::Kind::Dummy, jslot});
        x_col = X_realized_.col(jslot);
      } else if (cand->pool == Candidate::Pool::Real) {
        const int j = cand->index;
        actives_.push_back(j);
        is_active_[j] = 1;
        active_features_.push_back({ActiveFeature::Kind::Real, j});
        x_col = X_.col(j);
      } else {
        // Candidate::Pool::RealizedDummy should normally have is_new=false.
        const int jslot = cand->index;
        active_features_.push_back({ActiveFeature::Kind::Dummy, jslot});
        x_col = X_realized_.col(jslot);
      }

      // Store selected column in X_active_ for IRLS.
      const int k_new = static_cast<int>(active_features_.size()) - 1;
      X_active_.col(k_new) = x_col;

      // The active set changed, so the GLM fit on the active set must be recomputed.
      irls_stale_ = true;
    }

    // AFS/GLM update: solve IRLS if needed, blend coefficients, update eta,
    // mu, and the centered Poisson score.
    afs_blend_();

    // Important VD order:
    //   1. score has just changed;
    //   2. reveal/append the current score direction to the VD basis;
    //   3. then refresh all correlations using the expanded basis.
    // This must happen every iteration, not only when cand->is_new, because
    // repeated AFS shrinkage steps can change the score without changing the
    // active set.
    grow_basis_from_score_();
    full_corr_refresh_();

    record();

    if (T_realized_ > prev_T_realized && T_realized_ >= T) {
      step_ = it + 1;
      break;
    }

    step_ = it + 1;
  }

  const int cols = static_cast<int>(path.size());
  MatC out(p_, std::max(cols, 1));
  if (cols == 0) {
    out.col(0).setZero();
    return out;
  }

  for (int m = 0; m < cols; ++m)
    out.col(m) = path[m];

  return out;
}
