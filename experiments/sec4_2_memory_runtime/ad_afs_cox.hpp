// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// experiments/sec4_2_memory_runtime/ad_afs_cox.hpp
//
// AD_AFS_Cox -- Augmented-Dummy Adaptive Forward Stepwise with a Cox
// proportional-hazards link. C++ port of experiments/sec4_1_equivalence/AD_AFS_Cox.py, written as
// the explicit-augmentation counterpart of src/vd_afs_cox.cpp for the
// memory/runtime benchmark.
//
// Fabian's TRexSelector C++ library covers the linear model only, so there is
// no upstream Cox baseline to benchmark against. This is it.
//
// Same deliberate difference from the Python reference as the other three
// ports: X and D are kept as SEPARATE column-major maps instead of a
// concatenated [X | D], so the benchmark measures the algorithm and not an
// n*(p+L) copy. Candidate indices run 0..p-1 over X and p..p+L-1 over D.
//
// Everything else mirrors the Python line for line:
//   - y carries the non-negative event/censoring TIMES; the censoring indicator
//     delta (1 = event, 0 = right-censored) is a separate length-n array
//   - NO intercept: any constant added to eta cancels inside the risk-set
//     ratio, so the baseline hazard already plays that role. The active design
//     is the active columns alone, not [1 | active].
//   - score = martingale residual  s_i = delta_i - exp(eta_i) * H0_hat(t_i),
//     H0_hat the Breslow (or Efron) baseline cumulative hazard. Under Breslow
//     ties the residuals sum to exactly zero; the mean is removed defensively.
//   - selection on |corr| with corr = [X D]^T s  (single score, no across-class
//     aggregation -- unlike the multinomial port)
//   - Cox Newton-Raphson refit on the active set: gradient = score of the
//     partial log-likelihood, Hessian = observed information, both accumulated
//     over risk sets in one descending-time sweep, with an eps ridge on the
//     diagonal and a warm start from the current coefficients
//   - AFS blend: beta <- (1-rho) beta,  beta[A] += rho nu.  rho = 1 is GLM-OMP.
//   - stop once T dummies have entered
//   - run() returns the per-step path of REAL coefficients (p x steps), the
//     same object VD_AFS_Cox::run() returns, so both sides pay for it.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

class AD_AFS_Cox {
public:
  using MatC = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using Vec  = Eigen::VectorXd;

  struct Options {
    double rho             = 1.0;   // 1.0 = GLM-OMP, <1 damped AFS
    int    newton_max_iter = 25;
    double newton_tol      = 1e-8;
    double eps             = 1e-12;
    bool   efron           = false; // false = Breslow ties (default)
    int    max_steps       = -1;    // <0 -> p + T, matching VD_AFS_Cox::run
  };

  // X: n x p column-major, D: n x L column-major, y: event/censoring TIMES,
  // delta: length-n censoring indicator (nullptr -> all samples are events).
  // Columns of X and D are assumed centered and unit-L2 (the harness does it).
  AD_AFS_Cox(const double* Xptr, int n, int p,
             const double* Dptr, int L,
             const double* yptr, const double* deltaptr,
             const Options& o)
      : n_(n), p_(p), L_(L), opt_(o),
        X_(Xptr, n, p), D_(Dptr, n, L), y_(yptr, n) {
    delta_.resize(n_);
    for (int i = 0; i < n_; ++i)
      delta_(i) = (deltaptr && deltaptr[i] > 0.5) ? 1.0 : (deltaptr ? 0.0 : 1.0);
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
  int n_events()             const noexcept { return n_events_; }

private:
  static constexpr double kEtaClamp = 30.0;   // matches VD_AFS_Cox

  int n_, p_, L_;
  Options opt_;

  Eigen::Map<const MatC> X_, D_;
  Eigen::Map<const Vec>  y_;      // event/censoring times

  Vec delta_, eta_, w_, score_, beta_, corr_, nu_;

  std::vector<int> ord_desc_;     // sample indices, DESCENDING time
  int              n_events_ = 0;

  std::vector<int>  actives_;     // global indices: <p_ real, >=p_ dummy
  std::vector<char> is_active_;
  MatC Xa_;                       // n x capacity; NO intercept column
  int  k_ = 0;
  bool newton_stale_ = true;
  int  T_selected_ = 0, step_ = 0;

  // Newton scratch, allocated once and reused
  Vec  theta_, gvec_, stepv_, eta_a_, a_, mean1_, num1_, SR1_, SD1_, sumD_;
  MatC H_, SR2_, SD2_, num2_;

  static double clamp_eta_(double e) {
    return std::min(std::max(e, -kEtaClamp), kEtaClamp);
  }

  void init_() {
    n_events_ = 0;
    for (int i = 0; i < n_; ++i) if (delta_(i) > 0.5) ++n_events_;

    // Sample indices sorted by DESCENDING time; stable, so tie groups keep the
    // input order exactly as numpy's kind="stable" argsort does.
    ord_desc_.resize(n_);
    std::iota(ord_desc_.begin(), ord_desc_.end(), 0);
    std::stable_sort(ord_desc_.begin(), ord_desc_.end(),
                     [this](int a, int b) { return y_(a) > y_(b); });

    // Null model: eta = 0, w = 1.
    eta_ = Vec::Zero(n_);
    w_   = Vec::Ones(n_);
    update_score_();

    beta_ = Vec::Zero(p_ + L_);
    corr_.resize(p_ + L_);
    is_active_.assign(static_cast<std::size_t>(p_) + L_, 0);

    Xa_ = MatC::Zero(n_, 16);

    full_corr_refresh_();
  }

  // Martingale-residual score for a linear predictor eta. Walk samples in
  // descending time, group ties, accumulate the running risk-set sum SR0, form
  // per-distinct-time A_tau / B_tau, then
  //   H0(t_i) = sum_{tau >= position} A_tau  (accumulated in ascending time)
  //   s_i = delta_i - w_i * ( H0(t_i) - 1{event at t_i} * (A - B) ).
  void compute_score_(const Vec& eta, Vec& out) const {
    Vec w(n_);
    for (int i = 0; i < n_; ++i) w(i) = std::exp(clamp_eta_(eta(i)));

    std::vector<double> dtA, dtB;
    std::vector<int> grp(n_, 0);
    dtA.reserve(n_); dtB.reserve(n_);

    double SR0 = 0.0;
    int g = 0, idx = 0;
    while (idx < n_) {
      const double tau = y_(ord_desc_[idx]);
      int j = idx; double SD0 = 0.0; int d = 0;
      while (j < n_ && y_(ord_desc_[j]) == tau) {
        const int s = ord_desc_[j];
        SR0 += w(s);
        grp[s] = g;
        if (delta_(s) > 0.5) { SD0 += w(s); ++d; }
        ++j;
      }
      double A = 0.0, B = 0.0;
      if (d > 0) {
        if (!opt_.efron) {
          A = B = static_cast<double>(d) / SR0;              // Breslow
        } else {
          for (int l = 0; l < d; ++l) {                      // Efron
            const double frac = static_cast<double>(l) / d;
            const double denom = SR0 - frac * SD0;
            const double inv = (denom > 0.0) ? 1.0 / denom : 0.0;
            A += inv;
            B += (1.0 - frac) * inv;
          }
        }
      }
      dtA.push_back(A); dtB.push_back(B);
      idx = j; ++g;
    }

    const int G = g;
    std::vector<double> cumA(G);
    double run = 0.0;
    for (int gg = G - 1; gg >= 0; --gg) { run += dtA[gg]; cumA[gg] = run; }

    out.resize(n_);
    for (int i = 0; i < n_; ++i) {
      double H0 = cumA[grp[i]];
      if (delta_(i) > 0.5) H0 -= (dtA[grp[i]] - dtB[grp[i]]);  // own-time swap
      out(i) = delta_(i) - w(i) * H0;
    }
  }

  void update_score_() {
    for (int i = 0; i < n_; ++i) w_(i) = std::exp(clamp_eta_(eta_(i)));
    compute_score_(eta_, score_);
    // Sums to zero algebraically under Breslow ties; be defensive.
    score_.array() -= score_.mean();
  }

  void full_corr_refresh_() {
    corr_.head(p_).noalias() = X_.transpose() * score_;
    corr_.tail(L_).noalias() = D_.transpose() * score_;
  }

  // Cox gradient g (k) and observed information H (k x k) at eta, on the active
  // design A. Same risk-set sweep as the score, extended to the first (SR1) and
  // second (SR2) moments of the rows of A.
  void cox_grad_hess_(const Eigen::Ref<const MatC>& A, const Vec& eta,
                      Vec& g, MatC& H) {
    const int k = static_cast<int>(A.cols());
    g.setZero(k);
    H.setZero(k, k);
    if (k == 0) return;

    Vec w(n_);
    for (int i = 0; i < n_; ++i) w(i) = std::exp(clamp_eta_(eta(i)));

    double SR0 = 0.0;
    SR1_.setZero(k); SR2_.setZero(k, k);
    SD1_.resize(k);  SD2_.resize(k, k);
    sumD_.resize(k); a_.resize(k);
    mean1_.resize(k); num1_.resize(k); num2_.resize(k, k);

    int idx = 0;
    while (idx < n_) {
      const double tau = y_(ord_desc_[idx]);
      int j = idx; double SD0 = 0.0; int d = 0;
      SD1_.setZero(); SD2_.setZero(); sumD_.setZero();

      while (j < n_ && y_(ord_desc_[j]) == tau) {
        const int s = ord_desc_[j];
        a_ = A.row(s).transpose();
        const double ws = w(s);
        SR0 += ws;
        SR1_.noalias() += ws * a_;
        SR2_.noalias() += ws * (a_ * a_.transpose());
        if (delta_(s) > 0.5) {
          SD0 += ws;
          SD1_.noalias() += ws * a_;
          SD2_.noalias() += ws * (a_ * a_.transpose());
          sumD_ += a_;
          ++d;
        }
        ++j;
      }

      if (d > 0) {
        g += sumD_;
        if (!opt_.efron) {                                   // Breslow
          const double inv = 1.0 / SR0;
          mean1_ = SR1_ * inv;
          g.noalias() -= static_cast<double>(d) * mean1_;
          H.noalias() += static_cast<double>(d)
                       * (SR2_ * inv - mean1_ * mean1_.transpose());
        } else {                                             // Efron
          for (int l = 0; l < d; ++l) {
            const double frac = static_cast<double>(l) / d;
            const double denom = SR0 - frac * SD0;
            if (denom <= 0.0) continue;
            const double inv = 1.0 / denom;
            num1_ = SR1_ - frac * SD1_;
            num2_ = SR2_ - frac * SD2_;
            mean1_ = num1_ * inv;
            g.noalias() -= mean1_;
            H.noalias() += num2_ * inv - mean1_ * mean1_.transpose();
          }
        }
      }
      idx = j;
    }
  }

  void reserve_active_(int need) {
    if (Xa_.cols() >= static_cast<Eigen::Index>(need)) return;
    Eigen::Index cap = std::max<Eigen::Index>(16, Xa_.cols() * 2);
    while (cap < static_cast<Eigen::Index>(need)) cap *= 2;
    MatC tmp = MatC::Zero(n_, cap);
    tmp.leftCols(Xa_.cols()) = Xa_;
    Xa_.swap(tmp);
  }

  void add_active_(int j) {
    reserve_active_(k_ + 1);
    if (j < p_) Xa_.col(k_) = X_.col(j);
    else        Xa_.col(k_) = D_.col(j - p_);

    actives_.push_back(j);
    is_active_[j] = 1;
    if (j >= p_) ++T_selected_;
    ++k_;
    newton_stale_ = true;
  }

  void newton_solve_() {
    if (k_ == 0) { nu_.resize(0); newton_stale_ = false; return; }

    theta_.resize(k_);
    for (int i = 0; i < k_; ++i) theta_(i) = beta_(actives_[i]);   // warm start

    const auto A = Xa_.leftCols(k_);

    for (int iter = 0; iter < opt_.newton_max_iter; ++iter) {
      eta_a_.noalias() = A * theta_;
      cox_grad_hess_(A, eta_a_, gvec_, H_);
      H_.diagonal().array() += opt_.eps;                           // ridge

      Eigen::LDLT<MatC> ldlt(H_);
      if (ldlt.info() != Eigen::Success) break;
      stepv_ = ldlt.solve(gvec_);                    // Newton: theta += H^{-1} g
      if (!stepv_.allFinite()) break;

      theta_ += stepv_;
      if (stepv_.squaredNorm()
          < opt_.newton_tol * opt_.newton_tol * (1.0 + theta_.squaredNorm()))
        break;
    }

    nu_ = theta_;
    newton_stale_ = false;
  }

  void afs_blend_() {
    if (k_ == 0) return;
    if (newton_stale_) newton_solve_();

    beta_ *= (1.0 - opt_.rho);
    for (int i = 0; i < k_; ++i) beta_(actives_[i]) += opt_.rho * nu_(i);

    Vec ba(k_);
    for (int i = 0; i < k_; ++i) ba(i) = beta_(actives_[i]);

    eta_.noalias() = Xa_.leftCols(k_) * ba;          // no intercept
    update_score_();
    full_corr_refresh_();
  }
};
