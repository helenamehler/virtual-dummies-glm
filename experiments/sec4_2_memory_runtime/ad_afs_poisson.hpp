// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// experiments/sec4_2_memory_runtime/ad_afs_poisson.hpp
//
// AD_AFS_Poisson -- Augmented-Dummy Adaptive Forward Stepwise with a Poisson
// (log) link. C++ port of experiments/sec4_1_equivalence/AD_AFS_Poisson.py, written as the
// explicit-augmentation counterpart of src/vd_afs_poisson.cpp for the
// memory/runtime benchmark.
//
// Fabian's TRexSelector C++ library covers the linear model only
// (trex::tsolvers::linear_model::...), so there is no upstream Poisson
// baseline to benchmark against. This is it.
//
// The one deliberate difference from the Python reference: X and D are kept as
// SEPARATE column-major maps instead of a concatenated [X | D]. Concatenating
// would allocate a second n*(p+L) block and double the peak RSS, which would
// make the benchmark measure the copy rather than the algorithm. Candidate
// indices run 0..p-1 over X and p..p+L-1 over D, so selection, the active set
// and the IRLS refit behave exactly as on the augmented matrix.
//
// Everything else mirrors the Python line for line:
//   - counts y_count = max(0, y), raw, never centered
//   - null intercept b0 = log(mean(y_count)), the Poisson log-link MLE
//   - mu = exp(clamp(eta, +-30)), score s = y_count - mu, then CENTERED
//   - selection on |corr| with corr = [X D]^T s
//   - IRLS on [1 | active columns]: w = max(mu, 1e-10),
//     z = eta + (y - mu)/w, normal equations on the sqrt(w)-scaled design with
//     an eps ridge on the diagonal, warm-started from the current parameters
//   - AFS blend: b0 <- (1-rho) b0 + rho b0_irls;  beta <- (1-rho) beta,
//     beta[A] += rho nu.  rho = 1 is GLM-OMP.
//   - stop once T dummies have entered
//   - run() returns the per-step path of REAL coefficients (p x steps), the
//     same object VD_AFS_Poisson::run() returns, so both sides pay for it.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

class AD_AFS_Poisson {
public:
  using MatC = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using Vec  = Eigen::VectorXd;

  struct Options {
    double rho           = 1.0;   // 1.0 = GLM-OMP, <1 damped AFS
    int    irls_max_iter = 25;
    double irls_tol      = 1e-8;
    double eps           = 1e-12;
    int    max_steps     = -1;    // <0 -> p + T, matching VD_AFS_Poisson::run
  };

  // X: n x p column-major, D: n x L column-major, y: raw non-negative counts.
  // Columns of X and D are assumed centered and unit-L2 (the harness does it).
  AD_AFS_Poisson(const double* Xptr, int n, int p,
                 const double* Dptr, int L,
                 const double* yptr,
                 const Options& o)
      : n_(n), p_(p), L_(L), opt_(o),
        X_(Xptr, n, p), D_(Dptr, n, L), y_(yptr, n) {
    init_();
  }

  MatC run(int T) {
    std::vector<Vec> path;
    auto record = [&]() { path.emplace_back(beta_.head(p_)); };

    if (step_ == 0) record();

    const int max_steps = (opt_.max_steps > 0) ? opt_.max_steps : (p_ + T);

    for (int it = step_; it < max_steps; ++it) {
      const int prev_T = T_selected_;

      Eigen::Index jbest = 0;
      const double best = corr_.cwiseAbs().maxCoeff(&jbest);
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

private:
  static constexpr double kEtaClamp = 30.0;   // matches VD_AFS_Poisson
  static constexpr double kWFloor   = 1e-10;  // floor on the IRLS weights

  int n_, p_, L_;
  Options opt_;

  Eigen::Map<const MatC> X_, D_;
  Eigen::Map<const Vec>  y_;

  Vec y_count_, eta_, mu_, score_, beta_, corr_, nu_;
  double b0_ = 0.0, b0_irls_ = 0.0;

  std::vector<int>  actives_;    // global indices: <p_ real, >=p_ dummy
  std::vector<char> is_active_;
  MatC Xa_;                      // n x (1 + capacity); column 0 is the intercept
  int  k_ = 0;
  bool irls_stale_ = true;
  int  T_selected_ = 0, step_ = 0;

  // IRLS scratch, allocated once and reused
  MatC Xw_, G_;
  Vec  theta_, theta_new_, eta_a_, mu_a_, w_, sw_, z_, zw_, rhs_, ba_;

  static Vec mean_fn_(const Vec& e) {
    return e.array().min(kEtaClamp).max(-kEtaClamp).exp().matrix();
  }

  void init_() {
    y_count_ = y_.cwiseMax(0.0);

    const double ybar = std::max(y_count_.mean(), 1e-8);
    b0_      = std::log(ybar);
    b0_irls_ = b0_;
    eta_     = Vec::Constant(n_, b0_);
    update_score_();

    beta_ = Vec::Zero(p_ + L_);
    corr_.resize(p_ + L_);
    is_active_.assign(static_cast<std::size_t>(p_) + L_, 0);

    Xa_ = MatC::Zero(n_, 16);
    Xa_.col(0).setOnes();

    full_corr_refresh_();
  }

  void update_score_() {
    mu_    = mean_fn_(eta_);
    score_ = y_count_ - mu_;
    // The design lives in the centered subspace; with an intercept and AFS
    // blending the raw score is not exactly mean-zero at intermediate steps.
    score_.array() -= score_.mean();
  }

  void full_corr_refresh_() {
    corr_.head(p_).noalias() = X_.transpose() * score_;
    corr_.tail(L_).noalias() = D_.transpose() * score_;
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
    const int m = k_ + 1;

    theta_.resize(m);
    theta_(0) = b0_;
    for (int i = 0; i < k_; ++i) theta_(i + 1) = beta_(actives_[i]);

    const auto Xa = Xa_.leftCols(m);

    Xw_.resize(n_, m);
    G_.resize(m, m);

    for (int iter = 0; iter < opt_.irls_max_iter; ++iter) {
      eta_a_.noalias() = Xa * theta_;
      mu_a_ = mean_fn_(eta_a_);
      w_    = mu_a_.cwiseMax(kWFloor);
      z_    = eta_a_ + (y_count_ - mu_a_).cwiseQuotient(w_);
      sw_   = w_.cwiseSqrt();

      Xw_ = Xa.array().colwise() * sw_.array();
      zw_ = z_.cwiseProduct(sw_);

      G_.noalias() = Xw_.transpose() * Xw_;
      G_.diagonal().array() += opt_.eps;
      rhs_.noalias() = Xw_.transpose() * zw_;

      Eigen::LDLT<MatC> ldlt(G_);
      if (ldlt.info() != Eigen::Success) break;
      theta_new_ = ldlt.solve(rhs_);
      if (!theta_new_.allFinite()) break;

      const double delta = (theta_new_ - theta_).squaredNorm();
      theta_ = theta_new_;
      if (delta < opt_.irls_tol * opt_.irls_tol * (1.0 + theta_.squaredNorm()))
        break;
    }

    b0_irls_ = theta_(0);
    nu_      = theta_.tail(k_);
    irls_stale_ = false;
  }

  void afs_blend_() {
    if (k_ == 0) return;
    if (irls_stale_) irls_solve_();

    b0_ = (1.0 - opt_.rho) * b0_ + opt_.rho * b0_irls_;
    beta_ *= (1.0 - opt_.rho);
    for (int i = 0; i < k_; ++i) beta_(actives_[i]) += opt_.rho * nu_(i);

    ba_.resize(k_);
    for (int i = 0; i < k_; ++i) ba_(i) = beta_(actives_[i]);

    eta_.noalias() = Xa_.block(0, 1, n_, k_) * ba_;
    eta_.array() += b0_;

    update_score_();
    full_corr_refresh_();
  }
};
