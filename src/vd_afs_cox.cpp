// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// vd_afs_cox.cpp — GLM-style forward stepwise with the Cox proportional-hazards
// partial likelihood. Selection score = martingale residuals; active-set refit =
// Cox Newton-Raphson. No intercept (the baseline hazard absorbs it).
#include "vd_afs_cox.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>

// ============================================================================
// Martingale-residual score for a given linear predictor eta.
//
//   w_i        = exp(eta_i)
//   R(tau)     = { j : t_j >= tau }                       (risk set)
//   D(tau)     = { j : t_j = tau, delta_j = 1 }           (tied events)
//   Breslow :  A_tau = B_tau = d_tau / SR0(tau)
//   Efron   :  A_tau = sum_{l=0}^{d-1} 1 / (SR0 - (l/d) SD0)
//              B_tau = sum_{l=0}^{d-1} (1 - l/d) / (SR0 - (l/d) SD0)
//   H0(t_i)   = sum_{tau <= t_i} A_tau            (baseline cumulative hazard)
//   s_i       = delta_i - w_i * [ H0(t_i) - 1{event at t_i} * (A_{t_i}-B_{t_i}) ]
//
// For ANY eta this construction satisfies sum_i s_i = 0 exactly (algebraically),
// so the score lives in the centered subspace H by design.
// ============================================================================
void VD_AFS_Cox::compute_score_(const Vec& eta, Vec& score_out) const {
  const int n = n_;
  score_out.resize(n);

  Vec w(n);
  for (int i = 0; i < n; ++i) {
    double e = std::min(kEtaClamp, std::max(-kEtaClamp, eta(i)));
    w(i) = std::exp(e);
  }

  // Walk samples in DESCENDING time, grouping ties. As each group is added to
  // the running risk-set sum SR0, it becomes R(tau) for that group's time.
  std::vector<double> dt_A;      // A_tau per distinct time (descending order)
  std::vector<double> dt_B;      // B_tau per distinct time
  std::vector<int>    grp_of(n); // distinct-time group id per sample

  double SR0 = 0.0;
  int g = 0;
  int idx = 0;
  const int N = (int)ord_desc_.size();
  while (idx < N) {
    const double tau = time_(ord_desc_[idx]);
    int j = idx;
    double SD0 = 0.0;   // event-weight sum in this tie group
    int    d   = 0;     // number of events in this tie group
    // First pass over the group: add every member to the risk set.
    while (j < N && time_(ord_desc_[j]) == tau) {
      const int s = ord_desc_[j];
      SR0 += w(s);
      grp_of[s] = g;
      if (delta_(s) > 0.5) { SD0 += w(s); ++d; }
      ++j;
    }
    double A = 0.0, B = 0.0;
    if (d > 0) {
      if (!efron_) {
        A = B = double(d) / SR0;                 // Breslow
      } else {
        for (int l = 0; l < d; ++l) {            // Efron
          const double frac = double(l) / double(d);
          const double denom = SR0 - frac * SD0;
          const double inv = (denom > 0.0) ? 1.0 / denom : 0.0;
          A += inv;
          B += (1.0 - frac) * inv;
        }
      }
    }
    dt_A.push_back(A);
    dt_B.push_back(B);
    idx = j;
    ++g;
  }
  const int G = g;

  // Cumulative baseline hazard, accumulated in ASCENDING time (groups were
  // stored descending, so iterate from the last group to the first).
  std::vector<double> cumA(G, 0.0);
  double run = 0.0;
  for (int gg = G - 1; gg >= 0; --gg) {
    run += dt_A[gg];
    cumA[gg] = run;
  }

  for (int i = 0; i < n; ++i) {
    const int gi = grp_of[i];
    double H0 = cumA[gi];
    if (delta_(i) > 0.5) H0 -= (dt_A[gi] - dt_B[gi]);   // own-time Efron swap
    score_out(i) = delta_(i) - w(i) * H0;
  }
}

// ---------- Refresh w_, score_ from the current eta_ ----------
void VD_AFS_Cox::update_score_() {
  for (int i = 0; i < n_; ++i) {
    double e = std::min(kEtaClamp, std::max(-kEtaClamp, eta_(i)));
    w_(i) = std::exp(e);
  }
  compute_score_(eta_, score_);
  // The score sums to zero algebraically; subtract the tiny fp mean defensively
  // so it stays inside H exactly.
  score_.array() -= score_.mean();
  residuals_ = score_;   // base-class alias used by realize_dummy_'s provisional corr
  mu_ = w_;
}

// ============================================================================
// Cox Newton-Raphson gradient g (k) and observed information H (k x k) on the
// active design A at linear predictor eta. Same risk-set accumulation as the
// score, extended to first (SR1) and second (SR2) moments of the rows of A.
// ============================================================================
void VD_AFS_Cox::cox_grad_hess_(const MatC& A, const Vec& eta,
                                Vec& g, MatC& H) const {
  const int n = n_;
  const int k = (int)A.cols();
  g = Vec::Zero(k);
  H = MatC::Zero(k, k);
  if (k == 0) return;

  Vec w(n);
  for (int i = 0; i < n; ++i) {
    double e = std::min(kEtaClamp, std::max(-kEtaClamp, eta(i)));
    w(i) = std::exp(e);
  }

  double SR0 = 0.0;
  Vec  SR1 = Vec::Zero(k);
  MatC SR2 = MatC::Zero(k, k);

  const int N = (int)ord_desc_.size();
  int idx = 0;
  while (idx < N) {
    const double tau = time_(ord_desc_[idx]);
    int j = idx;

    // Group event moments D(tau).
    double SD0 = 0.0;
    Vec    SD1 = Vec::Zero(k);
    MatC   SD2 = MatC::Zero(k, k);
    Vec    sumD_a = Vec::Zero(k);   // sum of a_j over events (linear grad term)
    int    d = 0;

    while (j < N && time_(ord_desc_[j]) == tau) {
      const int s = ord_desc_[j];
      const auto a = A.row(s).transpose();      // k-vector
      const double ws = w(s);
      SR0 += ws;
      SR1.noalias() += ws * a;
      SR2.noalias() += ws * (a * a.transpose());
      if (delta_(s) > 0.5) {
        SD0 += ws;
        SD1.noalias() += ws * a;
        SD2.noalias() += ws * (a * a.transpose());
        sumD_a.noalias() += a;
        ++d;
      }
      ++j;
    }

    if (d > 0) {
      g.noalias() += sumD_a;   // + sum_{j in D} a_j

      if (!efron_) {           // Breslow
        const double inv = 1.0 / SR0;
        const Vec mean1 = SR1 * inv;
        g.noalias() -= double(d) * mean1;
        H.noalias() += double(d) * (SR2 * inv - mean1 * mean1.transpose());
      } else {                 // Efron
        for (int l = 0; l < d; ++l) {
          const double frac = double(l) / double(d);
          const double denom = SR0 - frac * SD0;
          if (denom <= 0.0) continue;
          const double inv = 1.0 / denom;
          const Vec  num1 = SR1 - frac * SD1;
          const MatC num2 = SR2 - frac * SD2;
          const Vec  mean1 = num1 * inv;
          g.noalias() -= mean1;
          H.noalias() += num2 * inv - mean1 * mean1.transpose();
        }
      }
    }
    idx = j;
  }
}

// ---------- Init ----------
void VD_AFS_Cox::init_cox_() {
  if (cox_inited_) return;
  rho_   = std::max(1e-6, std::min(opt_.rho, 1.0));
  efron_ = opt_.cox_efron;

  // y carries the times; delta comes from the separate cox_status array.
  time_ = y_;
  delta_.resize(n_);
  if ((int)opt_.cox_status.size() == n_) {
    for (int i = 0; i < n_; ++i) delta_(i) = (opt_.cox_status(i) > 0.5) ? 1.0 : 0.0;
  } else {
    delta_.setOnes();   // no censoring info => treat all samples as events
  }
  n_events_ = (int)std::llround(delta_.sum());

  // Sort sample indices by DESCENDING time (once; times never change).
  ord_desc_.resize(n_);
  std::iota(ord_desc_.begin(), ord_desc_.end(), 0);
  std::sort(ord_desc_.begin(), ord_desc_.end(),
            [&](int a, int b) { return time_(a) > time_(b); });

  // Null model: eta = 0, w = 1. Score = null martingale residuals.
  eta_ = Vec::Zero(n_);
  w_   = Vec::Ones(n_);
  update_score_();

  // VD_Base seeded the first basis direction from raw y_ (= time). The Cox
  // selector scores dummies with the martingale residual, so rebuild the VD
  // basis/projections from the centered null score instead.
  basis_indices_.clear();
  vd_rows_filled_ = 0;
  vd_stick_ = Vec::Ones(L_);
  vd_corr_  = Vec::Zero(L_);
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

  full_corr_refresh_();   // corr_ / vd_corr_ / corr_realized_ from the score

  nu_active_.resize(0);
  newton_stale_ = true;
  cox_inited_ = true;
}

// ---------- Candidate search (identical rule to the other AFS GLMs) ----------
std::optional<VD_AFS_Cox::Candidate>
VD_AFS_Cox::find_best_candidate_() const {
  Candidate best{Candidate::Pool::Real, -1, 0.0, true};

  for (int d = 0; d < L_; ++d) {
    if (vd_is_realized_[d]) continue;
    const double ac = std::abs(vd_corr_(d));
    if (ac > best.abs_corr) best = {Candidate::Pool::VD, d, ac, true};
  }
  for (int j = 0; j < T_realized_; ++j) {
    const double ac = std::abs(corr_realized_(j));
    if (ac > best.abs_corr)
      best = {Candidate::Pool::RealizedDummy, j, ac, false};
  }
  for (int j = 0; j < p_; ++j) {
    const double ac = std::abs(corr_(j));
    if (ac > best.abs_corr)
      best = {Candidate::Pool::Real, j, ac, !is_active_[j]};
  }

  if (best.index < 0 || best.abs_corr < 100.0 * opt_.eps)
    return std::nullopt;
  return best;
}

// ---------- Cox Newton-Raphson on the active set ----------
void VD_AFS_Cox::newton_solve_(int max_iter, double tol) {
  const int k = (int)active_features_.size();
  nu_active_.resize(k);
  if (k == 0) { newton_stale_ = false; return; }

  // Active design A = X_active columns. No intercept column (baseline hazard).
  MatC A(n_, k);
  for (int j = 0; j < k; ++j) A.col(j) = X_active_.col(j);

  // Warm start from the current blended coefficients.
  Vec theta(k);
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Real)
      theta(i) = beta_(af.index);
    else if (af.index >= 0 && af.index < beta_dummy_.size())
      theta(i) = beta_dummy_(af.index);
    else
      theta(i) = 0.0;
  }

  Vec  g;
  MatC H;
  for (int iter = 0; iter < max_iter; ++iter) {
    const Vec eta_a = A * theta;
    cox_grad_hess_(A, eta_a, g, H);
    H.diagonal().array() += opt_.eps;   // ridge for numerical stability

    Eigen::LLT<MatC> llt(H);
    if (llt.info() != Eigen::Success) break;   // keep current theta
    const Vec step = llt.solve(g);             // Newton: theta += H^{-1} g
    theta += step;

    if (step.squaredNorm() < tol * tol * (1.0 + theta.squaredNorm())) break;
  }

  nu_active_ = theta;
  newton_stale_ = false;
}

// ---------- AFS blend ----------
void VD_AFS_Cox::afs_blend_() {
  const int k = (int)active_features_.size();
  if (k == 0) return;
  if (newton_stale_) newton_solve_();

  // Blend real coefficients: beta <- (1-rho) beta_old + rho nu.
  beta_ *= (1.0 - rho_);
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Real)
      beta_(af.index) += rho_ * nu_active_(i);
  }
  // Blend realized-dummy coefficients.
  beta_dummy_.head(std::max(T_realized_, 1)) *= (1.0 - rho_);
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Dummy && af.index < beta_dummy_.size())
      beta_dummy_(af.index) += rho_ * nu_active_(i);
  }

  // Recompute eta from the blended coefficients (no intercept).
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

  update_score_();   // refresh w_, centered martingale score
}

// ---------- Run ----------
MatC VD_AFS_Cox::run(int T) {
  init_cox_();

  std::vector<Vec> path;
  auto record = [&]() {
    Vec s = beta_;
    if (normx_.size() == beta_.size()) s.array() /= normx_.array();
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
        const int jslot = cand->index;
        active_features_.push_back({ActiveFeature::Kind::Dummy, jslot});
        x_col = X_realized_.col(jslot);
      }

      const int k_new = (int)active_features_.size() - 1;
      if (k_new >= X_active_.cols())          // grow on demand (safety)
        X_active_.conservativeResize(n_, k_new + 16);
      X_active_.col(k_new) = x_col;
      newton_stale_ = true;
    }

    afs_blend_();

    // Order: score just changed -> reveal it to the VD basis -> refresh corrs.
    grow_basis_from_score_();
    full_corr_refresh_();

    record();

    if (T_realized_ > prev && T_realized_ >= T) { step_ = it + 1; break; }
    step_ = it + 1;
  }

  const int cols = (int)path.size();
  MatC out(p_, std::max(cols, 1));
  if (cols == 0) { out.col(0).setZero(); return out; }
  for (int m = 0; m < cols; ++m) out.col(m) = path[m];
  return out;
}
