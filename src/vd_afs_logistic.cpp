// UPSTREAM, EXTENDED: taulantkoka/virtual-dummies (Koka et al., 2026), modified for GLM/Cox score-based selection (Mehler, Koka, Muma, 2026). GPLv3, see NOTICE.
// vd_afs_logistic.cpp — GLM forward stepwise with logistic link.
#include "vd_afs_logistic.hpp"

// ---------- Sigmoid ----------
Vec VD_AFS_Logistic::sigmoid_(const Vec& eta) {
  return (1.0 + (-eta.array()).exp()).inverse().matrix();
}

// ---------- Update score from current eta ----------
void VD_AFS_Logistic::update_score_() {
  prob_ = sigmoid_(eta_);
  score_ = y_binary_ - prob_;
  score_.array() -= score_.mean();

  mu_ = prob_;
  residuals_ = score_;
}

// ---------- Init ----------
void VD_AFS_Logistic::init_logistic_() {
  if (logistic_inited_) return;
  rho_ = std::max(1e-6, std::min(opt_.rho, 1.0));

  y_binary_.resize(n_);
  for (int i = 0; i < n_; ++i)
    y_binary_(i) = (y_(i) > 0.0) ? 1.0 : 0.0;

  // Intercept-only null model: sigmoid(b0) = mean(y) = pbar
  const double pbar = std::clamp(y_binary_.mean(), 1e-6, 1.0 - 1e-6);
  b0_ = std::log(pbar / (1.0 - pbar));   // logit(pbar)
  b0_irls_ = b0_;
  eta_ = Vec::Constant(n_, b0_);

  // Sets prob_, centered score_, mu_, residuals_.
  update_score_();

  // VD_Base seeded the first basis direction from raw y_. The logistic selector
  // uses the score y - sigmoid(eta), so rebuild the initial VD basis from the
  // centered null score instead.
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

  // corr_ / vd_corr_ / corr_realized_ from the score.
  full_corr_refresh_();

  nu_active_.resize(0);
  irls_stale_ = true;
  logistic_inited_ = true;
}

// ---------- Candidate search (same as AFS: scan ALL features) ----------
std::optional<VD_AFS_Logistic::Candidate>
VD_AFS_Logistic::find_best_candidate_() const {
  Candidate best{Candidate::Pool::Real, -1, 0.0, true};

  // 1. Check Dummies first to be conservative
  for (int d = 0; d < L_; ++d) {
    if (vd_is_realized_[d]) continue;
    double ac = std::abs(vd_corr_(d));
    if (ac > best.abs_corr) best = {Candidate::Pool::VD, d, ac, true};
  }

  // 2. Check Realized Dummies (also nulls)
  for (int j = 0; j < T_realized_; ++j) {
    double ac = std::abs(corr_realized_(j));
    if (ac > best.abs_corr)
      best = {Candidate::Pool::RealizedDummy, j, ac, false};
  }

  // 3. Check Real features LAST
  // If a real feature has exactly the same correlation as a dummy, 
  // the '>' will fail, and the dummy will remain the 'best'.
  for (int j = 0; j < p_; ++j) {
    double ac = std::abs(corr_(j));
    if (ac > best.abs_corr) best = {Candidate::Pool::Real, j, ac, !is_active_[j]};
  }

  if (best.index < 0 || best.abs_corr < 100.0 * opt_.eps)
    return std::nullopt;
  return best;
}

// ---------- IRLS solve on active set ----------
void VD_AFS_Logistic::irls_solve_(int max_iter, double tol) {
  const int k  = (int)active_features_.size();
  const int kk = k + 1;   // +1 for the always-on intercept column

  Vec theta(kk);
  theta(0) = b0_;
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Real)
      theta(i + 1) = beta_(af.index);
    else if (af.index >= 0 && af.index < beta_dummy_.size())
      theta(i + 1) = beta_dummy_(af.index);
    else
      theta(i + 1) = 0.0;
  }

  // Augmented design [1 | X_active].
  MatC Xa(n_, kk);
  Xa.col(0).setOnes();
  for (int j = 0; j < k; ++j)
    Xa.col(j + 1) = X_active_.col(j);

  const double clamp_lo = 1e-10;
  const double clamp_hi = 1.0 - 1e-10;

  for (int iter = 0; iter < max_iter; ++iter) {
    Vec eta_a = Xa * theta;

    Vec p_a = sigmoid_(eta_a);
    for (int i = 0; i < n_; ++i)
      p_a(i) = std::clamp(p_a(i), clamp_lo, clamp_hi);

    Vec w = p_a.array() * (1.0 - p_a.array());
    Vec z = eta_a.array() + (y_binary_.array() - p_a.array()) / w.array();

    Vec sqrt_w = w.cwiseSqrt();
    MatC Xw(n_, kk);
    for (int j = 0; j < kk; ++j)
      Xw.col(j) = Xa.col(j).cwiseProduct(sqrt_w);
    Vec zw = z.cwiseProduct(sqrt_w);

    MatC G = Xw.transpose() * Xw;
    Vec rhs = Xw.transpose() * zw;
    G.diagonal().array() += opt_.eps;

    Eigen::LLT<MatC> llt(G);
    if (llt.info() != Eigen::Success) break;
    Vec theta_new = llt.solve(rhs);

    double delta = (theta_new - theta).squaredNorm();
    theta = theta_new;
    if (delta < tol * tol * (1.0 + theta.squaredNorm())) break;
  }

  b0_irls_ = theta(0);
  nu_active_.resize(k);
  for (int i = 0; i < k; ++i)
    nu_active_(i) = theta(i + 1);

  irls_stale_ = false;
}

// ---------- AFS blend ----------
void VD_AFS_Logistic::afs_blend_() {
  const int k = (int)active_features_.size();
  if (k == 0) return;
  if (irls_stale_) irls_solve_();

  // Blend intercept
  b0_ = (1.0 - rho_) * b0_ + rho_ * b0_irls_;

  // Blend coefficients: beta = (1-rho)*beta_old + rho*nu
  beta_ *= (1.0 - rho_);
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Real)
      beta_(af.index) += rho_ * nu_active_(i);
  }
  beta_dummy_.head(std::max(T_realized_, 1)) *= (1.0 - rho_);
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Dummy && af.index < beta_dummy_.size())
      beta_dummy_(af.index) += rho_ * nu_active_(i);
  }

  // Recompute eta from blended beta + intercept
  Vec beta_active(k);
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Real)
      beta_active(i) = beta_(af.index);
    else if (af.index < beta_dummy_.size())
      beta_active(i) = beta_dummy_(af.index);
    else
      beta_active(i) = 0.0;
  }
  eta_ = X_active_.leftCols(k) * beta_active;
  eta_.array() += b0_;               // <-- Intercept in eta

  update_score_();                   // ersetzt die alten mu_/residuals_-Zeilen
}
// ---------- Run ----------
MatC VD_AFS_Logistic::run(int T) {
  init_logistic_();

  std::vector<Vec> path;
  auto record = [&]() {
    Vec s = beta_;
    if (normx_.size() == beta_.size())
      s.array() /= normx_.array();
    path.emplace_back(std::move(s));
  };

  if (step_ == 0) record();
  const int max_steps = p_ + opt_.T_stop;

  for (int it = step_; it < max_steps; ++it) {
    const int prev = T_realized_;
    auto cand = find_best_candidate_();
    if (!cand) break;

    if (cand->is_new) {
      Vec x_col;
      if (cand->pool == Candidate::Pool::VD) {
        realize_dummy_(cand->index);
        int jslot = T_realized_ - 1;
        active_features_.push_back({ActiveFeature::Kind::Dummy, jslot});
        x_col = X_realized_.col(jslot);
      } else if (cand->pool == Candidate::Pool::Real) {
        int j = cand->index;
        actives_.push_back(j);
        is_active_[j] = 1;
        active_features_.push_back({ActiveFeature::Kind::Real, j});
        x_col = X_.col(j);
      } else {
        int jslot = cand->index;
        active_features_.push_back({ActiveFeature::Kind::Dummy, jslot});
        x_col = X_realized_.col(jslot);
      }

      // Store column in X_active_ for IRLS
      const int k_new = (int)active_features_.size() - 1;
      X_active_.col(k_new) = x_col;

      irls_stale_ = true;
    }

    // Blend (calls IRLS if stale)
    afs_blend_();

    // Grow basis from score direction (only when new feature entered)

    grow_basis_from_score_();

    full_corr_refresh_();
    record();

    if (T_realized_ > prev && T_realized_ >= T) { step_ = it + 1; break; }
    step_ = it + 1;
  }

  int cols = (int)path.size();
  MatC out(p_, std::max(cols, 1));
  if (cols == 0) { out.col(0).setZero(); return out; }
  for (int m = 0; m < cols; ++m) out.col(m) = path[m];
  return out;
}
