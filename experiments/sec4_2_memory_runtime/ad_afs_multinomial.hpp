// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// experiments/sec4_2_memory_runtime/ad_afs_multinomial.hpp
//
// AD_AFS_Multinomial -- Augmented-Dummy Adaptive Forward Stepwise with a
// reference-category multinomial (softmax) link. C++ port of
// experiments/sec4_1_equivalence/AD_AFS_Multinomial.py, written as the explicit-augmentation
// counterpart of src/vd_afs_multinomial.cpp for the memory/runtime
// benchmark.
//
// Fabian's TRexSelector C++ library covers the linear model only, so there is
// no upstream multinomial baseline to benchmark against. This is it.
//
// Same deliberate difference from the Python reference as the Poisson and
// logistic ports: X and D are kept as SEPARATE column-major maps instead of a
// concatenated [X | D], so the benchmark measures the algorithm and not an
// n*(p+L) copy. Candidate indices run 0..p-1 over X and p..p+L-1 over D.
//
// Everything else mirrors the Python line for line:
//   - class 0 is the reference; Cm = C-1 non-reference score directions
//     s^(c) = y^(c) - p^(c), p = softmax_ref(eta), each column CENTERED
//   - null intercepts b0_c = log(p_c / p_0) from the class marginals
//   - selection score is the L2 norm across ALL C classes of the per-class
//     inner products, recovered from the Cm tracked ones via the zero-sum
//     identity  ||g_j||^2 = ||G_j||^2 + (sum_c G_jc)^2, so the ranking does not
//     depend on the reference class; matches class_score_norm() /
//     full_corr_refresh_multi_() in the C++ VD solver
//   - reference-category multinomial Newton on [1 | active columns]: parameter
//     matrix Theta is (kk x Cm), kk = k+1, flattened CLASS-MAJOR into a dense
//     M x M Fisher system (M = kk*Cm),
//         H block (c, c') = Xa^T diag( p_c ([c==c'] - p_c') ) Xa,
//         g block  (c)    = Xa^T ( y^(c) - p^(c) ),
//     with an eps ridge on the diagonal, warm-started from the current Theta
//   - AFS blend: b0 <- (1-rho) b0 + rho b0_irls;  Beta <- (1-rho) Beta,
//     Beta[A] += rho Nu.  rho = 1 is GLM-OMP.
//   - the per-feature path scalar is the across-class L2 norm of the sum-to-zero
//     representative of the per-class coefficients (class_coef_norm_), exactly
//     what VD_AFS_Multinomial::run() records
//   - stop once T dummies have entered
//
// NOTE on labels: y carries the RAW integer class labels 0..C-1, not a centered
// response. C is inferred as max(y)+1, the same rule as init_multinomial_().
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

class AD_AFS_Multinomial {
public:
  using MatC = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using Vec  = Eigen::VectorXd;

  struct Options {
    double rho           = 1.0;   // 1.0 = GLM-OMP, <1 damped AFS
    int    irls_max_iter = 25;
    double irls_tol      = 1e-8;
    double eps           = 1e-12;
    int    max_steps     = -1;    // <0 -> p + T, matching VD_AFS_Multinomial::run
  };

  // X: n x p column-major, D: n x L column-major, y: raw integer labels 0..C-1.
  // Columns of X and D are assumed centered and unit-L2 (the harness does it).
  AD_AFS_Multinomial(const double* Xptr, int n, int p,
                     const double* Dptr, int L,
                     const double* yptr,
                     const Options& o)
      : n_(n), p_(p), L_(L), opt_(o),
        X_(Xptr, n, p), D_(Dptr, n, L), y_(yptr, n) {
    init_();
  }

  MatC run(int T) {
    std::vector<Vec> path;
    auto record = [&]() {
      path.emplace_back(class_coef_norm_(beta_multi_.topRows(p_), C_));
    };

    if (step_ == 0) record();

    const int max_steps = (opt_.max_steps > 0) ? opt_.max_steps : (p_ + T);

    for (int it = step_; it < max_steps; ++it) {
      const int prev_T = T_selected_;

      Eigen::Index jbest = 0;
      const double best = corr_.maxCoeff(&jbest);   // corr_ >= 0 by construction
      if (best < 100.0 * opt_.eps) break;

      const int j = static_cast<int>(jbest);
      if (!is_active_[j]) add_active_(j);

      afs_blend_();
      record();
      step_ = it + 1;

      if (T_selected_ > prev_T && T_selected_ >= T) break;
    }

    const int cols = static_cast<int>(path.size());
    MatC out(p_, std::max(cols, 1));
    if (cols == 0) {
      out.col(0).setZero();
      return out;
    }
    for (int m = 0; m < cols; ++m) out.col(m) = path[m];
    return out;
  }

  int num_selected_dummies() const noexcept { return T_selected_; }
  int steps()                const noexcept { return step_; }
  int active_size()          const noexcept { return k_; }
  int basis_size()           const noexcept { return k_; }  // for symmetry
  int n_classes()            const noexcept { return C_; }

private:
  static constexpr double kEtaClamp = 30.0;   // matches VD_AFS_Multinomial

  int n_, p_, L_;
  Options opt_;

  Eigen::Map<const MatC> X_, D_;
  Eigen::Map<const Vec>  y_;

  // ---- Reference-invariant across-class aggregation ----
  // The full C-class score vector sums to zero, so the held-out entry is
  //   g_j^(0) = -sum_{c>=1} G_jc   and   ||g_j||^2 = ||G_j||^2 + (sum_c G_jc)^2.
  // Invariant to relabeling the classes; the truncated ||G_j|| is not.
  // Mirrors class_score_norm() in src/vd_afs_multinomial.cpp.
  static Vec class_score_norm_(const MatC& G) {
    const Vec q = G.rowwise().squaredNorm();
    const Vec t = G.rowwise().sum();
    return (q.array() + t.array().square()).sqrt().matrix();
  }
  // Softmax coefficients are identified only up to a shift along the class
  // axis; the sum-to-zero representative of row B_j (with b_j^(0) = 0) has
  //   ||b_j - mean_c(b_j)||^2 = ||B_j||^2 - (sum_c B_jc)^2 / C.
  static Vec class_coef_norm_(const MatC& B, int C) {
    const Vec q = B.rowwise().squaredNorm();
    const Vec t = B.rowwise().sum();
    return (q.array() - t.array().square() / double(C)).max(0.0).sqrt().matrix();
  }

  int C_ = 0, Cm_ = 0;

  MatC Y_ind_;        // n x Cm  indicators for classes 1..C-1
  MatC eta_;          // n x Cm
  MatC prob_;         // n x Cm
  MatC score_;        // n x Cm, each column centered
  MatC beta_multi_;   // (p+L) x Cm
  Vec  b0_, b0_irls_; // Cm
  Vec  corr_;         // p+L, across-class L2 norm

  std::vector<int>  actives_;    // global indices: <p_ real, >=p_ dummy
  std::vector<char> is_active_;
  MatC Xa_;                      // n x (1 + capacity); column 0 is the intercept
  MatC Nu_;                      // k x Cm, IRLS solution on the active set
  int  k_ = 0;
  bool irls_stale_ = true;
  int  T_selected_ = 0, step_ = 0;

  // Newton scratch, allocated once and reused
  MatC Theta_, eta_a_, Pa_, H_, B_, XaW_, CorrBlock_;
  Vec  g_, delta_, w_, resid_;

  // Reference-category softmax: p_ic = exp(eta_ic) / (1 + sum_c' exp(eta_ic')).
  void softmax_(const MatC& e, MatC& out) const {
    out = e.array().min(kEtaClamp).max(-kEtaClamp).exp().matrix();
    for (int i = 0; i < out.rows(); ++i) {
      const double denom = 1.0 + out.row(i).sum();
      out.row(i) /= denom;
    }
  }

  void init_() {
    double ymax = 0.0;
    for (int i = 0; i < n_; ++i) ymax = std::max(ymax, y_(i));
    C_  = std::max(2, static_cast<int>(std::llround(ymax)) + 1);
    Cm_ = C_ - 1;

    Y_ind_ = MatC::Zero(n_, Cm_);
    for (int i = 0; i < n_; ++i) {
      const int lab = static_cast<int>(std::llround(y_(i)));
      if (lab >= 1 && lab <= Cm_) Y_ind_(i, lab - 1) = 1.0;
    }

    // Null intercepts: b0_c = log(p_c / p_0), class marginals.
    Vec counts = Y_ind_.colwise().sum().transpose();
    const double n0 = std::max(static_cast<double>(n_) - counts.sum(), 1e-8);
    const double p0 = std::max(n0 / static_cast<double>(n_), 1e-8);
    b0_.resize(Cm_);
    for (int c = 0; c < Cm_; ++c) {
      const double pc = std::max(counts(c) / static_cast<double>(n_), 1e-8);
      b0_(c) = std::log(pc / p0);
    }
    b0_irls_ = b0_;

    eta_ = MatC::Zero(n_, Cm_);
    eta_.rowwise() = b0_.transpose();
    update_score_();

    beta_multi_ = MatC::Zero(p_ + L_, Cm_);
    corr_.resize(p_ + L_);
    is_active_.assign(static_cast<std::size_t>(p_) + L_, 0);

    Xa_ = MatC::Zero(n_, 16);
    Xa_.col(0).setOnes();

    full_corr_refresh_();
  }

  void update_score_() {
    softmax_(eta_, prob_);
    score_ = Y_ind_ - prob_;
    // Project each class column back to the centered subspace H. With the
    // per-class intercepts at the MLE the column sums are exactly zero; under
    // AFS blending (rho < 1) they are only approximately zero.
    for (int c = 0; c < Cm_; ++c) score_.col(c).array() -= score_.col(c).mean();
  }

  // corr(j) = || ( <x_j, s^(c)> )_c ||_2 over ALL C class scores, recovered
  // from the Cm tracked ones (see class_score_norm_).
  void full_corr_refresh_() {
    CorrBlock_.noalias() = X_.transpose() * score_;          // p x Cm
    corr_.head(p_) = class_score_norm_(CorrBlock_);
    CorrBlock_.noalias() = D_.transpose() * score_;          // L x Cm
    corr_.tail(L_) = class_score_norm_(CorrBlock_);
  }

  void reserve_active_(int need) {  // need = number of active columns wanted
    if (Xa_.cols() >= static_cast<Eigen::Index>(need) + 1) return;
    Eigen::Index cap = std::max<Eigen::Index>(16, Xa_.cols() * 2);
    while (cap < static_cast<Eigen::Index>(need) + 1) cap *= 2;
    MatC tmp = MatC::Zero(n_, cap);
    tmp.leftCols(Xa_.cols()) = Xa_;
    Xa_.swap(tmp);
  }

  void add_active_(int j) {
    reserve_active_(k_ + 1);
    if (j < p_) Xa_.col(k_ + 1) = X_.col(j);
    else        Xa_.col(k_ + 1) = D_.col(j - p_);

    actives_.push_back(j);
    is_active_[j] = 1;
    if (j >= p_) ++T_selected_;
    ++k_;
    irls_stale_ = true;
  }

  void irls_solve_() {
    const int kk = k_ + 1;
    const int M  = kk * Cm_;

    const auto Xa = Xa_.leftCols(kk);

    // Warm start Theta from the current blended values.
    Theta_.resize(kk, Cm_);
    Theta_.row(0) = b0_.transpose();
    for (int i = 0; i < k_; ++i) Theta_.row(i + 1) = beta_multi_.row(actives_[i]);

    g_.resize(M);
    H_.resize(M, M);
    XaW_.resize(n_, kk);

    for (int iter = 0; iter < opt_.irls_max_iter; ++iter) {
      eta_a_.noalias() = Xa * Theta_;
      softmax_(eta_a_, Pa_);

      g_.setZero();
      H_.setZero();

      for (int c = 0; c < Cm_; ++c) {
        resid_ = Y_ind_.col(c) - Pa_.col(c);
        g_.segment(c * kk, kk).noalias() = Xa.transpose() * resid_;

        for (int cp = c; cp < Cm_; ++cp) {
          const double kron = (c == cp) ? 1.0 : 0.0;
          w_ = (Pa_.col(c).array() * (kron - Pa_.col(cp).array())).matrix();
          XaW_ = Xa.array().colwise() * w_.array();
          B_.noalias() = Xa.transpose() * XaW_;             // kk x kk
          H_.block(c * kk, cp * kk, kk, kk) = B_;
          if (cp != c) H_.block(cp * kk, c * kk, kk, kk) = B_.transpose();
        }
      }

      H_.diagonal().array() += opt_.eps;                     // ridge

      Eigen::LDLT<MatC> ldlt(H_);
      if (ldlt.info() != Eigen::Success) break;
      delta_ = ldlt.solve(g_);                               // Newton step
      if (!delta_.allFinite()) break;

      // delta_ is CLASS-MAJOR: block c is the update for Theta_.col(c).
      for (int c = 0; c < Cm_; ++c) Theta_.col(c) += delta_.segment(c * kk, kk);

      const double dn = delta_.squaredNorm();
      const double tn = Theta_.squaredNorm();
      if (dn < opt_.irls_tol * opt_.irls_tol * (1.0 + tn)) break;
    }

    b0_irls_ = Theta_.row(0).transpose();
    Nu_      = Theta_.bottomRows(k_);
    irls_stale_ = false;
  }

  void afs_blend_() {
    if (k_ == 0) return;
    if (irls_stale_) irls_solve_();

    b0_ = (1.0 - opt_.rho) * b0_ + opt_.rho * b0_irls_;
    beta_multi_ *= (1.0 - opt_.rho);
    for (int i = 0; i < k_; ++i)
      beta_multi_.row(actives_[i]) += opt_.rho * Nu_.row(i);

    MatC Ba(k_, Cm_);
    for (int i = 0; i < k_; ++i) Ba.row(i) = beta_multi_.row(actives_[i]);

    eta_.noalias() = Xa_.block(0, 1, n_, k_) * Ba;
    eta_.rowwise() += b0_.transpose();

    update_score_();
    full_corr_refresh_();
  }
};
