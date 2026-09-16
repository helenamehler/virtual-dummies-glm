// UPSTREAM, EXTENDED: taulantkoka/virtual-dummies (Koka et al., 2026), modified for GLM/Cox score-based selection (Mehler, Koka, Muma, 2026). GPLv3, see NOTICE.
// vd_common.hpp
// Shared types, options, and helpers for the VD-* family of solvers.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <optional>
#include <utility>
#include <unordered_set>
#include <limits>
#include <tuple>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#if !defined(_WIN32)
  #include <sys/mman.h>
  #include <unistd.h>
#endif

using Vec     = Eigen::VectorXd;
using MatC    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatR    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapMatC = Eigen::Map<const MatC>;
using MapVec  = Eigen::Map<const Vec>;

enum class VDDummyLaw : uint8_t { Spherical = 0, Gaussian = 1 };

struct VDOptions {
  int    T_stop       = 100;
  int    max_vd_proj = 100;
  double eps         = 1e-12;
  bool   standardize = false;
  bool   debug       = false;
  unsigned long long seed = 0ULL;
  VDDummyLaw dummy_law = VDDummyLaw::Spherical;
  double rho = 1.0;
  int mmap_fd         = -1;
  int mmap_block_cols = 0;

  // ---- Cox proportional-hazards extras ----
  // For VD_AFS_Cox the response y carries the (non-negative) event/censoring
  // TIMES. The censoring indicator delta (1 = event observed, 0 = censored)
  // is supplied here as a SEPARATE length-n array, so the uniform
  // (X, y, num_dummies, options) constructor is preserved. Ignored by every
  // non-Cox selector. If left empty, VD_AFS_Cox treats all samples as events.
  Vec  cox_status;          // length n, entries in {0,1}
  bool cox_efron = false;   // false = Breslow ties (default), true = Efron
};

static constexpr int VD_Y_SENTINEL     = -1;
static constexpr int VD_DUMMY_SENTINEL = -2;

struct ActiveFeature {
  enum class Kind : uint8_t { Real, Dummy };
  Kind kind;
  int  index;
};

namespace vd_detail {

inline int colblock() {
  static int B = [](){
    const char* s = std::getenv("VD_COLBLOCK");
    long v = s ? std::strtol(s, nullptr, 10) : 0;
    if (v <= 0) v = 8192;
    return int(v);
  }();
  return B;
}

#if !defined(_WIN32)
inline bool pread_full(int fd, void* buf, std::size_t len, off_t offset) {
  char* p = static_cast<char*>(buf);
  std::size_t remaining = len;
  while (remaining > 0) {
    ssize_t r = ::pread(fd, p, remaining, offset);
    if (r <= 0) return false;
    p += r; offset += r; remaining -= static_cast<std::size_t>(r);
  }
  return true;
}
#endif

inline void advise_sequential(const void* ptr, std::size_t len) {
#if !defined(_WIN32)
  if (ptr && len > 0)
    ::madvise(const_cast<void*>(ptr), len, MADV_RANDOM);
#else
  (void)ptr; (void)len;
#endif
}

inline void gemv_Xt(const MapMatC& X, const Vec& v, Vec& out,
                    int mmap_fd, int block_cols, double* scratch)
{
  const int n = (int)X.rows(), p = (int)X.cols();
  out.setZero(p);
#if !defined(_WIN32)
  const bool use_pread = (mmap_fd >= 0 && scratch && block_cols > 0);
#else
  const bool use_pread = false;
  (void)mmap_fd; (void)scratch;
#endif
  const int B = (block_cols > 0) ? block_cols : colblock();
  for (int j0 = 0; j0 < p; j0 += B) {
    const int jb = std::min(B, p - j0);
    const double* Xblk;
#if !defined(_WIN32)
    if (use_pread) {
      pread_full(mmap_fd, scratch, std::size_t(jb)*n*sizeof(double),
                 off_t(j0)*n*sizeof(double));
      Xblk = scratch;
    } else
#endif
    { Xblk = X.col(j0).data(); }
    for (int j = 0; j < jb; ++j) {
      const double* x = Xblk + std::ptrdiff_t(j)*n;
      double s = 0.0;
      for (int i = 0; i < n; ++i) s += x[i]*v[i];
      out[j0+j] = s;
    }
  }
}

inline void gemv_Xv(const MapMatC& X, const Vec& v, Vec& y,
                    int mmap_fd, int block_cols, double* scratch)
{
  const int n = (int)X.rows(), p = (int)X.cols();
  y.setZero(n);
#if !defined(_WIN32)
  const bool use_pread = (mmap_fd >= 0 && scratch && block_cols > 0);
#else
  const bool use_pread = false;
  (void)mmap_fd; (void)scratch;
#endif
  const int B = (block_cols > 0) ? block_cols : colblock();
  for (int j0 = 0; j0 < p; j0 += B) {
    const int jb = std::min(B, p - j0);
    const double* Xblk;
#if !defined(_WIN32)
    if (use_pread) {
      pread_full(mmap_fd, scratch, std::size_t(jb)*n*sizeof(double),
                 off_t(j0)*n*sizeof(double));
      Xblk = scratch;
    } else
#endif
    { Xblk = X.col(j0).data(); }
    for (int j = 0; j < jb; ++j) {
      const double* x = Xblk + std::ptrdiff_t(j)*n;
      const double w = v[j0+j];
      if (w == 0.0) continue;
      for (int i = 0; i < n; ++i) y[i] += x[i]*w;
    }
  }
}

// ---- Multi-RHS variant of gemv_Xt ------------------------------------------
// OUT(:,c) = X^T V(:,c) for every column of V, in ONE pass over X.
//
// Why this exists: the multinomial selector needs Cm = C-1 correlation vectors
// per step. Calling gemv_Xt once per class walks X Cm times, and -- measured,
// not assumed -- the cost is dominated by the serial dependency chain of the
// scalar accumulator, not by memory bandwidth. Keeping Cm independent
// accumulators in one inner loop breaks that chain Cm ways: ~2.7x on a 3-class
// problem. Traffic reduction is a secondary benefit.
//
// The accumulation order per (column, class) is IDENTICAL to gemv_Xt -- strictly
// increasing i, one accumulator per class -- so results are bit-for-bit the same
// as the per-class loop this replaces.
template <int M>
inline void gemv_Xt_multi_impl(const MapMatC& X, const double* const* vp,
                               MatC& OUT, int mmap_fd, int block_cols,
                               double* scratch)
{
  const int n = (int)X.rows(), p = (int)X.cols();
#if !defined(_WIN32)
  const bool use_pread = (mmap_fd >= 0 && scratch && block_cols > 0);
#else
  const bool use_pread = false;
  (void)mmap_fd; (void)scratch;
#endif
  const int B = (block_cols > 0) ? block_cols : colblock();
  for (int j0 = 0; j0 < p; j0 += B) {
    const int jb = std::min(B, p - j0);
    const double* Xblk;
#if !defined(_WIN32)
    if (use_pread) {
      pread_full(mmap_fd, scratch, std::size_t(jb)*n*sizeof(double),
                 off_t(j0)*n*sizeof(double));
      Xblk = scratch;
    } else
#endif
    { Xblk = X.col(j0).data(); }
    for (int j = 0; j < jb; ++j) {
      const double* x = Xblk + std::ptrdiff_t(j)*n;
      double s[M] = {0.0};
      for (int i = 0; i < n; ++i) {
        const double xi = x[i];
        for (int c = 0; c < M; ++c) s[c] += xi * vp[c][i];
      }
      for (int c = 0; c < M; ++c) OUT(j0+j, c) = s[c];
    }
  }
}

inline void gemv_Xt_multi(const MapMatC& X, const MatC& V, MatC& OUT,
                          int mmap_fd, int block_cols, double* scratch)
{
  const int p = (int)X.cols();
  const int m = (int)V.cols();
  OUT.setZero(p, std::max(m, 1));
  if (m <= 0) return;

  constexpr int kMaxRhs = 6;
  if (m > kMaxRhs) {                     // more classes than we specialize for
    Vec tmp(p), vc;
    for (int c = 0; c < m; ++c) {
      vc = V.col(c);
      gemv_Xt(X, vc, tmp, mmap_fd, block_cols, scratch);
      OUT.col(c) = tmp;
    }
    return;
  }

  const double* vp[kMaxRhs];
  for (int c = 0; c < m; ++c) vp[c] = V.col(c).data();

  switch (m) {
    case 1: gemv_Xt_multi_impl<1>(X, vp, OUT, mmap_fd, block_cols, scratch); break;
    case 2: gemv_Xt_multi_impl<2>(X, vp, OUT, mmap_fd, block_cols, scratch); break;
    case 3: gemv_Xt_multi_impl<3>(X, vp, OUT, mmap_fd, block_cols, scratch); break;
    case 4: gemv_Xt_multi_impl<4>(X, vp, OUT, mmap_fd, block_cols, scratch); break;
    case 5: gemv_Xt_multi_impl<5>(X, vp, OUT, mmap_fd, block_cols, scratch); break;
    default: gemv_Xt_multi_impl<6>(X, vp, OUT, mmap_fd, block_cols, scratch); break;
  }
}

inline void gemv_Xt(const MapMatC& X, const Vec& v, Vec& out) {
  gemv_Xt(X, v, out, -1, 0, nullptr);
}
inline void gemv_Xv(const MapMatC& X, const Vec& v, Vec& y) {
  gemv_Xv(X, v, y, -1, 0, nullptr);
}

inline double gamma_mt(double m, std::mt19937_64& rng) {
    auto uniform01 = [&](){
      constexpr double scale = 1.0/(1ULL<<53);
      return (rng()>>(64-53))*scale;
    };
    std::normal_distribution<double> normal(0.0,1.0);
    if (m < 1.0) {
      double g = gamma_mt(m+1.0, rng);
      double u = std::max(uniform01(), std::numeric_limits<double>::min());
      return g*std::pow(u, 1.0/m);
    }
    const double d=m-1.0/3.0, c=1.0/std::sqrt(9.0*d);
    for (;;) {
      double z=normal(rng), v=1.0+c*z;
      if (v<=0.0) continue;
      double v3=v*v*v;
      if (uniform01()<1.0-0.0331*(z*z)*(z*z)) return d*v3;
      double u=std::max(uniform01(), std::numeric_limits<double>::min());
      if (std::log(u)<0.5*z*z+d*(1.0-v3+std::log(v3))) return d*v3;
    }
}

} // namespace vd_detail
