// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// vd_afs_multinomial.cpp — GLM forward stepwise with multinomial (softmax) link.
#include "vd_afs_multinomial.hpp"

#include <algorithm>
#include <cmath>

// ---------- Reference-invariant across-class aggregation ----------
namespace {

// Rank statistic for a candidate whose tracked per-class score inner products
// are the row G_j = ( <x_j, s^(1)>, ..., <x_j, s^(C-1)> ).
//
// The full C-class score vector g_j = ( <x_j, s^(0)>, ..., <x_j, s^(C-1)> )
// sums to zero, since sum_{c=0}^{C-1} ( y_i^(c) - p_i^(c) ) = 0 for every i.
// The reference entry is therefore g_j^(0) = -sum_{c>=1} G_jc and
//     ||g_j||^2 = ||G_j||^2 + ( sum_c G_jc )^2 ,
// recoverable from the stored Cm columns alone. Relabeling the classes permutes
// the entries of g_j, so ||g_j||_2 is unchanged; the truncated ||G_j||_2 is not.
//
// VD compatibility is untouched: s^(0) is a linear combination of s^(1..C-1),
// which already span V_k, so <d, s^(0)> follows from the revealed projections.
// No extra basis direction, no extra stick-breaking draw, no extra memory.
inline Vec class_score_norm(const MatC& G) {
  const Vec q = G.rowwise().squaredNorm();
  const Vec t = G.rowwise().sum();
  return (q.array() + t.array().square()).sqrt().matrix();
}

// Same idea for the reported per-feature coefficient magnitude. Softmax
// coefficients are identified only up to a shift along the class axis; the
// sum-to-zero (symmetric) representative of row B_j, with b_j^(0) = 0, has
//     || b_j - mean_c(b_j) ||^2 = ||B_j||^2 - ( sum_c B_jc )^2 / C .
// Zero iff the feature never entered the active set, as before.
inline Vec class_coef_norm(const MatC& B, int C) {
  const Vec q = B.rowwise().squaredNorm();
  const Vec t = B.rowwise().sum();
  return (q.array() - t.array().square() / double(C)).max(0.0).sqrt().matrix();
}

}  // namespace

// ---------- Reference-category softmax ----------
// eta has Cm_ columns (non-reference classes); the reference class 0 has eta=0.
//   p_ic = exp(eta_ic) / (1 + sum_{c'} exp(eta_ic')),   c = 1..Cm_.
void VD_AFS_Multinomial::softmax_(const MatC& eta, MatC& prob) const {
  const int n = (int)eta.rows();
  prob.resize(n, Cm_);
  for (int i = 0; i < n; ++i) {
    double denom = 1.0;  // exp(0) contribution of the reference class
    for (int c = 0; c < Cm_; ++c) {
      double e = std::min(kEtaClamp, std::max(-kEtaClamp, eta(i, c)));
      double ex = std::exp(e);
      prob(i, c) = ex;
      denom += ex;
    }
    const double inv = 1.0 / denom;
    for (int c = 0; c < Cm_; ++c) prob(i, c) *= inv;
  }
}

// ---------- Update score from current eta ----------
void VD_AFS_Multinomial::update_score_() {
  softmax_(eta_, prob_);
  score_ = Y_ind_ - prob_;

  // Each class score must live in the centered subspace H. With per-class
  // intercepts at the MLE the column sums are exactly zero; under AFS blending
  // (rho<1) they are only approximately zero, so project every column back to H
  // before it is used for VD basis growth or correlations.
  for (int c = 0; c < Cm_; ++c)
    score_.col(c).array() -= score_.col(c).mean();

  // Keep base-class aliases loosely consistent (used only by realize_dummy_'s
  // provisional corr, which full_corr_refresh_multi_ overwrites).
  residuals_ = score_.col(0);
}

// ---------- Init ----------
void VD_AFS_Multinomial::init_multinomial_() {
  if (multinomial_inited_) return;
  rho_ = std::max(1e-6, std::min(opt_.rho, 1.0));

  // Infer the number of classes from the integer labels 0..C-1 in y_.
  double ymax = 0.0;
  for (int i = 0; i < n_; ++i) ymax = std::max(ymax, y_(i));
  C_  = std::max(2, (int)std::llround(ymax) + 1);
  Cm_ = C_ - 1;

  // Build {0,1} indicator columns for the non-reference classes 1..C-1.
  Y_ind_ = MatC::Zero(n_, Cm_);
  for (int i = 0; i < n_; ++i) {
    int lab = (int)std::llround(y_(i));
    if (lab >= 1 && lab <= Cm_) Y_ind_(i, lab - 1) = 1.0;
  }

  // Intercept-only null model. Null MLE: p_c = n_c / n (class marginal), so
  //   b0_c = log(p_c / p_0),  eta constant across rows.
  b0_.resize(Cm_);
  Vec pc(Cm_);
  double p0 = 0.0;
  {
    Vec counts = Y_ind_.colwise().sum();       // n_c for c=1..C-1
    double n0 = n_ - counts.sum();             // reference-class count
    p0 = std::max(n0 / n_, 1e-8);
    for (int c = 0; c < Cm_; ++c) {
      pc(c) = std::max(counts(c) / n_, 1e-8);
      b0_(c) = std::log(pc(c) / p0);
    }
  }
  b0_irls_ = b0_;
  eta_ = MatC::Zero(n_, Cm_);
  for (int c = 0; c < Cm_; ++c) eta_.col(c).setConstant(b0_(c));

  // Sets prob_, centered score_.
  update_score_();

  // VD_Base seeded the first basis direction from raw y_. The multinomial
  // selector uses the per-class scores, so rebuild the VD basis/projections
  // from the centered null scores instead.
  basis_indices_.clear();
  vd_rows_filled_ = 0;
  vd_stick_ = Vec::Ones(L_);
  vd_corr_ = Vec::Zero(L_);
  std::fill(vd_is_realized_.begin(), vd_is_realized_.end(), 0);
  vd_unrealized_idx_.clear();
  T_realized_ = 0;

  // Seed the basis with the first null-score direction, then draw the initial
  // dummy projections (stick-breaking row 0)...
  const double sn0 = score_.col(0).norm();
  if (sn0 > opt_.eps * std::sqrt(double(n_))) {
    y_norm_ = sn0;
    basis_.col(0) = score_.col(0) / sn0;
    basis_indices_.push_back(VD_Y_SENTINEL);
    initialize_virtual_dummies_();
  }
  // ...and append the remaining Cm_-1 null-score directions, drawing a fresh
  // projection row along each (update_virtual_dummies_).
  for (int c = 1; c < Cm_; ++c) {
    if (basis_size_() >= basis_.cols()) break;
    if (auto vo = orthonormalize_(score_.col(c))) {
      basis_.col(basis_size_()) = *vo;
      basis_indices_.push_back(VD_Y_SENTINEL);
      update_virtual_dummies_();
    }
  }

  full_corr_refresh_multi_();

  beta_multi_       = MatC::Zero(p_, Cm_);
  beta_dummy_multi_ = MatC::Zero(std::max(opt_.T_stop, 1), Cm_);
  nu_active_.resize(0, Cm_);
  irls_stale_ = true;
  multinomial_inited_ = true;
}

// ---------- Grow the basis from every class score direction ----------
// Appends each s^(c) that is not already spanned by V_k, drawing fresh dummy
// projections along each newly revealed axis. dim(V_k) can grow by up to Cm_
// per step; this is the (paper-sanctioned) cost of the multi-class score.
void VD_AFS_Multinomial::grow_basis_from_scores_() {
  for (int c = 0; c < Cm_; ++c) {
    if (basis_size_() >= basis_.cols()) return;
    if (auto vo = orthonormalize_(score_.col(c))) {
      basis_.col(basis_size_()) = *vo;
      basis_indices_.push_back(VD_DUMMY_SENTINEL);  // generic revealed direction
      update_virtual_dummies_();
    }
  }
}

// ---------- Per-class correlations aggregated to an L2 norm ----------
void VD_AFS_Multinomial::full_corr_refresh_multi_() {
  const int nb = basis_size_();
  const int m_rows = std::min(nb, vd_rows_filled_);

  // Real features: corr_(j) = || ( <x_j, s^(c)> )_c ||_2.
  // One pass over X for all Cm class scores. gemv_Xt_multi keeps Cm independent
  // accumulators in the inner loop; per (column, class) the summation order is
  // the same as the per-class gemv_Xt loop this replaces, so the result is
  // bit-identical -- it is only faster (see the comment on gemv_Xt_multi).
  MatC Greal(p_, Cm_);
  vd_detail::gemv_Xt_multi(X_, score_, Greal,
      opt_.mmap_fd, opt_.mmap_block_cols, scratch_ptr_());
  corr_ = class_score_norm(Greal);

  // Virtual dummies: g_d^(c) = A_k^T (E_k^T s^(c)); aggregate over classes.
  if (m_rows > 0 && L_ > 0) {
    MatC Gvd = MatC::Zero(L_, Cm_);
    for (int c = 0; c < Cm_; ++c) {
      Vec bp = basis_.leftCols(nb).transpose() * score_.col(c);
      Gvd.col(c).noalias() =
          vd_proj_.topRows(m_rows).transpose() * bp.head(m_rows);
    }
    const Vec gvd_norm = class_score_norm(Gvd);
    for (int d = 0; d < L_; ++d)
      vd_corr_(d) = vd_is_realized_[d] ? 0.0 : gvd_norm(d);
  }

  // Realized dummies (also null competitors).
  if (T_realized_ > 0) {
    MatC Grz = MatC::Zero(T_realized_, Cm_);
    for (int c = 0; c < Cm_; ++c)
      Grz.col(c).noalias() =
          X_realized_.leftCols(T_realized_).transpose() * score_.col(c);
    corr_realized_.head(T_realized_) = class_score_norm(Grz);
  }
}

// ---------- Candidate search (dummies first, strict '>' => conservative ties) ----------
std::optional<VD_AFS_Multinomial::Candidate>
VD_AFS_Multinomial::find_best_candidate_() const {
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

// ---------- Multinomial IRLS on [1 | X_active] ----------
// Reference-category Newton on the stacked softmax likelihood. Parameter matrix
// Theta is (kk x Cm_) with kk = k+1 (intercept + active features); flattened
// column-major (class-major blocks) into a dense M x M Fisher system,
//   Hessian block (c,c') = Xa^T diag( p_c (delta_cc' - p_c') ) Xa,
//   gradient block  (c)  = Xa^T ( y^(c) - p^(c) ).
void VD_AFS_Multinomial::irls_solve_(int max_iter, double tol) {
  const int k  = (int)active_features_.size();
  const int kk = k + 1;                 // +1 intercept
  const int M  = kk * Cm_;

  // Augmented design [1 | X_active].
  MatC Xa(n_, kk);
  Xa.col(0).setOnes();
  for (int j = 0; j < k; ++j) Xa.col(j + 1) = X_active_.col(j);

  // Warm start Theta from current blended values.
  MatC Theta(kk, Cm_);
  Theta.row(0) = b0_.transpose();
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    for (int c = 0; c < Cm_; ++c) {
      if (af.kind == ActiveFeature::Kind::Real)
        Theta(i + 1, c) = beta_multi_(af.index, c);
      else if (af.index >= 0 && af.index < beta_dummy_multi_.rows())
        Theta(i + 1, c) = beta_dummy_multi_(af.index, c);
      else
        Theta(i + 1, c) = 0.0;
    }
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    MatC eta_a = Xa * Theta;            // n x Cm_
    MatC Pa; softmax_(eta_a, Pa);       // n x Cm_

    // Gradient (M) and Fisher information (M x M).
    Vec g = Vec::Zero(M);
    MatC H = MatC::Zero(M, M);

    for (int c = 0; c < Cm_; ++c) {
      Vec resid = Y_ind_.col(c) - Pa.col(c);       // n
      g.segment(c * kk, kk).noalias() = Xa.transpose() * resid;

      for (int cp = c; cp < Cm_; ++cp) {
        // w_i = p_ic * ( [c==cp] - p_icp )
        Vec w(n_);
        for (int i = 0; i < n_; ++i)
          w(i) = Pa(i, c) * ((c == cp ? 1.0 : 0.0) - Pa(i, cp));
        MatC Xw = Xa.array().colwise() * w.array();   // n x kk
        MatC B  = Xa.transpose() * Xw;                // kk x kk
        H.block(c * kk, cp * kk, kk, kk) = B;
        if (cp != c) H.block(cp * kk, c * kk, kk, kk) = B.transpose();
      }
    }

    H.diagonal().array() += opt_.eps;   // ridge for numerical stability
    Eigen::LLT<MatC> llt(H);
    if (llt.info() != Eigen::Success) break;
    Vec delta = llt.solve(g);           // Newton step: Theta += H^{-1} g

    // Unflatten and update.
    double dn = 0.0, tn = 0.0;
    for (int c = 0; c < Cm_; ++c) {
      Theta.col(c) += delta.segment(c * kk, kk);
      dn += delta.segment(c * kk, kk).squaredNorm();
      tn += Theta.col(c).squaredNorm();
    }
    if (dn < tol * tol * (1.0 + tn)) break;
  }

  b0_irls_ = Theta.row(0).transpose();
  nu_active_.resize(k, Cm_);
  for (int i = 0; i < k; ++i) nu_active_.row(i) = Theta.row(i + 1);
  irls_stale_ = false;
}

// ---------- AFS blend ----------
void VD_AFS_Multinomial::afs_blend_() {
  const int k = (int)active_features_.size();
  if (k == 0) return;
  if (irls_stale_) irls_solve_();

  // Grow the realized-dummy coefficient buffer if more dummies were realized
  // than the initial T_stop guess.
  if (T_realized_ > beta_dummy_multi_.rows()) {
    MatC tmp = MatC::Zero(T_realized_ + 8, Cm_);
    tmp.topRows(beta_dummy_multi_.rows()) = beta_dummy_multi_;
    beta_dummy_multi_.swap(tmp);
  }

  // Blend intercepts and per-class coefficients: theta <- (1-rho) old + rho irls.
  b0_ = (1.0 - rho_) * b0_ + rho_ * b0_irls_;

  beta_multi_ *= (1.0 - rho_);
  const int ndum = std::min(std::max(T_realized_, 1), (int)beta_dummy_multi_.rows());
  beta_dummy_multi_.topRows(ndum) *= (1.0 - rho_);
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Real)
      beta_multi_.row(af.index) += rho_ * nu_active_.row(i);
    else if (af.index >= 0 && af.index < beta_dummy_multi_.rows())
      beta_dummy_multi_.row(af.index) += rho_ * nu_active_.row(i);
  }

  // Recompute eta from blended coefficients + intercepts.
  MatC nu_blend(k, Cm_);
  for (int i = 0; i < k; ++i) {
    const auto& af = active_features_[i];
    if (af.kind == ActiveFeature::Kind::Real)
      nu_blend.row(i) = beta_multi_.row(af.index);
    else if (af.index >= 0 && af.index < beta_dummy_multi_.rows())
      nu_blend.row(i) = beta_dummy_multi_.row(af.index);
    else
      nu_blend.row(i).setZero();
  }
  eta_ = X_active_.leftCols(k) * nu_blend;   // n x Cm_
  for (int c = 0; c < Cm_; ++c) eta_.col(c).array() += b0_(c);

  update_score_();
}

// ---------- Run ----------
MatC VD_AFS_Multinomial::run(int T) {
  init_multinomial_();

  std::vector<Vec> path;
  auto record = [&]() {
    // Representative per-feature magnitude, reference-invariant.
    Vec s = class_coef_norm(beta_multi_, C_);
    if (normx_.size() == s.size()) s.array() /= normx_.array();
    beta_ = s;
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
      // X_active_ is only sized to min(n,p) by the base class. With a multi-class
      // score, many reals can enter before T dummies are realized, so grow it on
      // demand to avoid a heap overflow.
      if (k_new >= X_active_.cols())
        X_active_.conservativeResize(n_, k_new + 16);
      X_active_.col(k_new) = x_col;
      irls_stale_ = true;
    }

    afs_blend_();

    // Order: score just changed -> reveal every new score axis to the VD basis
    // -> refresh all correlations against the expanded basis.
    grow_basis_from_scores_();
    full_corr_refresh_multi_();

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
