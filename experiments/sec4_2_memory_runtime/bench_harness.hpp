// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// experiments/sec4_2_memory_runtime/bench_harness.hpp
//
// Shared harness for the memory/runtime benchmarks. Everything in here is
// family-independent: core pinning, single-threaded BLAS, the fork/pipe/RSS
// measurement, the memory-cap check, the dummy-block generator and the CSV
// timestamp.
//
// The two older drivers (bench_memory_logistic.cpp for the Gaussian
// family and bench_memory_poisson.cpp) carry their own
// inline copy of this code and are deliberately left untouched -- they have
// already produced results. The logistic and multinomial drivers include this
// header instead, so the harness exists once rather than four times. If you
// ever change a measurement convention here, the older two do NOT follow.
#pragma once

#include <armadillo>

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <string>

#include <pthread.h>
#if defined(__APPLE__)
#include <sys/qos.h>
#endif

// ----------------------------------------------------
//  CORE PINNING
// ----------------------------------------------------
inline void pin_to_core(int core_id) {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        // Slurm cpusets rarely contain the system-wide id we ask for. Harmless.
        std::cerr << "\n[WARN] Failed to pin to core " << core_id << "\n";
    }
#elif defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    (void)core_id;
#else
    (void)core_id;
#endif
}

// ----------------------------------------------------
//  MEMORY-CAP HELPERS
// ----------------------------------------------------
inline double env_mem_cap_gb(const char* var) {
    const char* s = std::getenv(var);
    if (!s || !*s) return std::numeric_limits<double>::infinity();
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s || v <= 0.0) return std::numeric_limits<double>::infinity();
    return v;
}

// bytes needed ~ n * (p + L) * 8   (AD materializes X and D)
//               ~ n * p * 8        (VD: dummies stay virtual)
inline long double ad_bytes_needed(int n, int p, int L) {
    return 8.0L * (long double)n * (long double)(p + L);
}
inline long double vd_bytes_needed(int n, int p) {
    return 8.0L * (long double)n * (long double)p;
}

// Peak RSS in BYTES. Linux ru_maxrss is KiB, macOS is bytes.
inline uint64_t peak_rss_bytes() {
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__) && defined(__MACH__)
    return static_cast<uint64_t>(ru.ru_maxrss);
#else
    return static_cast<uint64_t>(ru.ru_maxrss) * 1024ULL;
#endif
}

inline double peak_rss_mib() {
    return static_cast<double>(peak_rss_bytes()) / (1024.0 * 1024.0);
}

inline void cap_blas_env() {
    setenv("OMP_NUM_THREADS",        "1",     1);
    setenv("OPENBLAS_NUM_THREADS",   "1",     1);
    setenv("MKL_NUM_THREADS",        "1",     1);
    setenv("VECLIB_MAXIMUM_THREADS", "1",     1);
    setenv("MKL_DYNAMIC",            "FALSE", 1);
    setenv("OMP_DYNAMIC",            "FALSE", 1);
    setenv("OMP_PROC_BIND",          "TRUE",  1);
}

inline void progress(int current, int total, int p, int m, int L, int rep) {
    double pct = 100.0 * current / total;
    std::cerr << "\r[ " << int(pct) << "% ]  "
              << "p=" << p << ", m=" << m << ", L=" << L << ", rep=" << rep
              << std::flush;
}

// ----------------------------------------------------
//  SAFE PIPE I/O
// ----------------------------------------------------
struct ChildResult {
    double sec;
    double peak_mb;   // MiB
    int    ok;
    int    code;      // 1 = refused by the memory cap, 2 = died
    int    steps;     // observed path length
    int    n_dummies; // dummies selected
};

inline void write_all(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    size_t left = n;
    while (left) {
        ssize_t w = ::write(fd, p, left);
        if (w < 0) { if (errno == EINTR) continue; _exit(111); }
        left -= static_cast<size_t>(w);
        p    += w;
    }
}

inline bool read_all_safe(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    size_t left = n;
    while (left) {
        ssize_t r = ::read(fd, p, left);
        if (r <= 0) return false;
        left -= static_cast<size_t>(r);
        p    += r;
    }
    return true;
}

// Refuse the run and report it, instead of letting the OOM killer decide.
inline bool refuse_if_over_cap(long double need_b, const char* cap_var,
                               long double safety, int out_fd) {
    const long double cap_b =
        (long double)env_mem_cap_gb(cap_var) * 1024.0L * 1024.0L * 1024.0L;
    if ((need_b * safety) > cap_b) {
        ChildResult out{};
        out.ok = 0; out.code = 1;
        out.peak_mb = peak_rss_mib();
        write_all(out_fd, &out, sizeof(out));
        return true;
    }
    return false;
}

// ----------------------------------------------------
//  run_in_child
// ----------------------------------------------------
template <typename Fn>
inline ChildResult run_in_child(Fn fn, int n, int p, int L, int T, uint64_t seed) {
    int fds[2];
    if (pipe(fds) != 0) std::abort();
    pid_t pid = fork();
    if (pid < 0) std::abort();

    if (pid == 0) {
        ::close(fds[0]);
        int out_fd = fds[1];
        if (out_fd != 3) {
            if (dup2(out_fd, 3) < 0) _exit(113);
            ::close(out_fd);
            out_fd = 3;
        }
        fn(n, p, L, T, seed, out_fd);
        _exit(0);
    }

    ::close(fds[1]);
    ChildResult r{};
    bool okread = read_all_safe(fds[0], &r, sizeof(r));
    ::close(fds[0]);

    int st = 0;
    (void)waitpid(pid, &st, 0);

    if (!okread) { r.sec = 0.0; r.peak_mb = 0.0; r.ok = 0; r.code = 2; return r; }
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) { r.ok = 0; if (r.code == 0) r.code = 2; }
    return r;
}

// ----------------------------------------------------
//  SHARED DATA-GEN PIECES
// ----------------------------------------------------
// Center each column and scale it to unit L2 norm, in place.
inline void center_unit_l2(arma::mat& M) {
    for (arma::uword j = 0; j < M.n_cols; ++j) {
        arma::vec c = M.col(j);
        c -= arma::mean(c);
        const double nrm = arma::norm(c, 2);
        if (nrm > 1e-12) c /= nrm;
        M.col(j) = c;
    }
}

// The explicit dummy block. Same construction and same seed offset as the
// Gaussian and Poisson drivers, so the AD side sees the dummies the VD side
// samples from the same null distribution.
inline void gen_dummies(int n, int L, uint64_t seed, arma::mat& D) {
    D.set_size(n, L);
    arma::arma_rng::set_seed(seed + 2);
    D.randn();
    center_unit_l2(D);
}

// ----------------------------------------------------
//  timestamp
// ----------------------------------------------------
inline std::string current_timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}
