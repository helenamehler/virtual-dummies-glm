// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// experiments/sec4_2_memory_runtime/bench_memory_poisson.cpp
//
// 2-way runtime/memory benchmark for the POISSON AFS family:
//   * VD-AFS-Poisson  (Eigen, src/vd_afs_poisson.hpp)      -> columns *_vd
//   * AD-AFS-Poisson  (Eigen, experiments/sec4_2_memory_runtime/ad_afs_poisson.hpp) -> columns *_ad
//
// Fabian's TRexSelector covers the linear model only, so the explicit-dummy
// baseline here is our own C++ port of experiments/sec4_1_equivalence/AD_AFS_Poisson.py. Both
// sides are Eigen, single-threaded, and run in separate forked processes on
// identical generated data, same harness as bench_memory_logistic.
//
// CLUSTER CONFIG (ICASSP memory point):
//   n = 10 000, p = 100 000, L = 5p = 500 000, T = 10, rho = 1.0, 20 replicates.
//   AD child: X and D at once, n*(p+L)*8 = 4.8e10 B = 44.7 GiB.
//   VD child: X only, n*p*8 = 8.0e9 B = 7.45 GiB, plus the k x L projections.
//   Request at least 64 GB and keep ADPOIS_MEM_CAP_GB above 58 (the code
//   multiplies the requirement by a 1.3 safety factor before comparing).
//
// Memory columns are MiB (bytes / 1024^2); meta.json records the unit.
// For the paper, convert with MiB * 1.048576 = MB.
#include <armadillo>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <fstream>
#include <limits>
#include <filesystem>
#include <random>
#include <ctime>

#include "vd_afs_poisson.hpp"   // VD side (existing)
#include "ad_afs_poisson.hpp"             // AD side (this paper)

#include <pthread.h>
#if defined(__APPLE__)
#include <sys/qos.h>
#endif

static void pin_to_core(int core_id) {
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
#endif
}

// ----------------------------------------------------
//  CONFIG
// ----------------------------------------------------
static const int N_FIXED   = 10000;
static const int T_SELECT  = 10;
static const int REPS      = 20;

static const int P_LIST[]  = {100000};
static const int NUM_P     = sizeof(P_LIST) / sizeof(P_LIST[0]);

static const int L_MULT[]  = {5};
static const int NUM_M     = sizeof(L_MULT) / sizeof(L_MULT[0]);

static const uint64_t BASE_SEED = 123;

static const int    K_ACTIVE  = 10;    // active predictors
static const double AMP_SD    = 0.5;   // sd of the log-mean linear predictor
static const double BASE_RATE = 2.0;   // E[lambda], i.e. mean count
static const double AFS_RHO   = 1.0;   // 1.0 = GLM-OMP (both solvers)

// ----------------------------------------------------
//  MEMORY-CAP HELPERS
// ----------------------------------------------------
static double env_mem_cap_gb(const char* var) {
    const char* s = std::getenv(var);
    if (!s || !*s) return std::numeric_limits<double>::infinity();
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s || v <= 0.0) return std::numeric_limits<double>::infinity();
    return v;
}

// bytes needed ~ n * (p + L) * 8   (AD materializes X and D)
//               ~ n * p * 8        (VD: dummies stay virtual)
static long double ad_bytes_needed(int n, int p, int L) {
    return 8.0L * (long double)n * (long double)(p + L);
}
static long double vd_bytes_needed(int n, int p) {
    return 8.0L * (long double)n * (long double)p;
}

static inline uint64_t peak_rss_bytes() {
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__) && defined(__MACH__)
    return static_cast<uint64_t>(ru.ru_maxrss);
#else
    return static_cast<uint64_t>(ru.ru_maxrss) * 1024ULL;
#endif
}

static void cap_blas_env() {
    setenv("OMP_NUM_THREADS",        "1",     1);
    setenv("OPENBLAS_NUM_THREADS",   "1",     1);
    setenv("MKL_NUM_THREADS",        "1",     1);
    setenv("VECLIB_MAXIMUM_THREADS", "1",     1);
    setenv("MKL_DYNAMIC",            "FALSE", 1);
    setenv("OMP_DYNAMIC",            "FALSE", 1);
    setenv("OMP_PROC_BIND",          "TRUE",  1);
}

static void progress(int current, int total, int p, int m, int L, int rep) {
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
    double peak_mb;
    int    ok;
    int    code;
    int    steps;     // observed path length, sets k
    int    n_dummies; // dummies selected
};

static void write_all(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    size_t left = n;
    while (left) {
        ssize_t w = ::write(fd, p, left);
        if (w < 0) { if (errno == EINTR) continue; _exit(111); }
        left -= static_cast<size_t>(w);
        p    += w;
    }
}

static bool read_all_safe(int fd, void* buf, size_t n) {
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

// ----------------------------------------------------
//  DATA GEN — Poisson counts, shared by both children
// ----------------------------------------------------
//
// Mirrors experiments/sec4_1_equivalence/sim_snr_poisson.py:
//   eta_lin = X * beta_base, rescaled to sd = AMP_SD
//   intercept via log-mean-exp so that E[lambda] = BASE_RATE
//   y ~ Poisson(exp(eta_lin + intercept))
// The rescaling makes the signal independent of n and of the unit-L2 column
// scaling, so the same AMP_SD means the same thing at every problem size.
static void gen_problem_poisson(int n, int p, uint64_t seed,
                                arma::mat& X, arma::vec& y)
{
    arma::arma_rng::set_seed(seed);
    X = arma::randn(n, p);
    for (arma::uword j = 0; j < static_cast<arma::uword>(p); ++j) {
        arma::vec c = X.col(j);
        c -= arma::mean(c);
        double nrm = arma::norm(c, 2);
        if (nrm > 1e-12) c /= nrm;
        X.col(j) = c;
    }

    std::mt19937_64 rng(seed + 7);
    std::uniform_int_distribution<int> coin(0, 1);

    arma::vec beta_base(p, arma::fill::zeros);
    const int s = std::min(K_ACTIVE, p);
    for (int j = 0; j < s; ++j) beta_base(j) = coin(rng) ? 1.0 : -1.0;

    arma::vec eta = X * beta_base;
    eta -= arma::mean(eta);
    const double sd = arma::stddev(eta);
    if (sd > 1e-12) eta *= (AMP_SD / sd);

    // intercept so that mean(exp(eta + b0)) == BASE_RATE, via log-mean-exp
    const double mx  = eta.max();
    const double lme = mx + std::log(arma::mean(arma::exp(eta - mx)) + 1e-300);
    const double b0  = std::log(BASE_RATE) - lme;

    y.set_size(n);
    for (int i = 0; i < n; ++i) {
        const double lam = std::exp(eta(i) + b0);
        std::poisson_distribution<long long> pois(lam);
        y(i) = static_cast<double>(pois(rng));
    }
    // NOTE: no centering. The Poisson solvers consume RAW counts.
}

static void gen_dummies(int n, int L, uint64_t seed, arma::mat& D)
{
    D.set_size(n, L);
    arma::arma_rng::set_seed(seed + 2);
    D.randn();
    for (arma::uword j = 0; j < static_cast<arma::uword>(L); ++j) {
        arma::vec c = D.col(j);
        c -= arma::mean(c);
        double nrm = arma::norm(c, 2);
        if (nrm > 1e-12) c /= nrm;
        D.col(j) = c;
    }
}

// ----------------------------------------------------
//  CHILD WORKERS
// ----------------------------------------------------
static void child_vd_pois(int n, int p, int L, int T,
                          uint64_t seed, int out_fd) {
    pin_to_core(2);
    cap_blas_env();
    ChildResult out{}; out.ok = 1; out.code = 0;

    long double need_b = vd_bytes_needed(n, p);
    long double cap_b  = env_mem_cap_gb("VD_MEM_CAP_GB") * 1024LL * 1024LL * 1024LL;
    if (need_b > cap_b) {
        out.ok = 0; out.code = 1;
        out.peak_mb = static_cast<double>(peak_rss_bytes()) / (1024.0 * 1024.0);
        write_all(out_fd, &out, sizeof(out)); _exit(0);
    }

    arma::mat X; arma::vec y;
    gen_problem_poisson(n, p, seed, X, y);

    using MatCol = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    Eigen::Map<const MatCol>        Xeig(X.memptr(), n, p);
    Eigen::Map<const Eigen::VectorXd> Yeig(y.memptr(), n);

    VDOptions opt;
    opt.T_stop      = T;
    opt.max_vd_proj = n;
    opt.standardize = false;
    opt.rho         = AFS_RHO;
    opt.seed        = seed;

    VD_AFS_Poisson vd(Xeig.data(), n, p, Yeig.data(), n, L, opt);

    auto t0 = std::chrono::high_resolution_clock::now();
    (void)vd.run(T);
    auto t1 = std::chrono::high_resolution_clock::now();

    out.sec       = std::chrono::duration<double>(t1 - t0).count();
    out.peak_mb   = static_cast<double>(peak_rss_bytes()) / (1024.0 * 1024.0);
    // Path length, i.e. one entry per selection step (real or dummy). NOT
    // basis_size(): that counts the orthogonal projection basis, which also
    // holds the score direction and one row per revealed dummy direction, so
    // it runs well above the step count and is not comparable to AD's step_.
    out.steps     = static_cast<int>(vd.active_features_copy().size());
    out.n_dummies = vd.num_realized_dummies();
    write_all(out_fd, &out, sizeof(out)); _exit(0);
}

static void child_ad_pois(int n, int p, int L, int T,
                          uint64_t seed, int out_fd) {
    pin_to_core(2);
    cap_blas_env();
    ChildResult out{}; out.ok = 1; out.code = 0;

    long double need_b = ad_bytes_needed(n, p, L);
    long double cap_b  = env_mem_cap_gb("ADPOIS_MEM_CAP_GB") * 1024LL * 1024LL * 1024LL;
    static const long double MEM_SAFETY = 1.3L;
    if ((need_b * MEM_SAFETY) > cap_b) {
        out.ok = 0; out.code = 1;
        out.peak_mb = static_cast<double>(peak_rss_bytes()) / (1024.0 * 1024.0);
        write_all(out_fd, &out, sizeof(out)); _exit(0);
    }

    arma::mat X; arma::vec y;
    gen_problem_poisson(n, p, seed, X, y);

    arma::mat D;
    gen_dummies(n, L, seed, D);

    AD_AFS_Poisson::Options o;
    o.rho = AFS_RHO;

    // X and D stay separate: concatenating would double the peak RSS and the
    // benchmark would measure the copy instead of the algorithm.
    AD_AFS_Poisson ad(X.memptr(), n, p, D.memptr(), L, y.memptr(), o);

    auto t0 = std::chrono::high_resolution_clock::now();
    (void)ad.run(T);
    auto t1 = std::chrono::high_resolution_clock::now();

    out.sec       = std::chrono::duration<double>(t1 - t0).count();
    out.peak_mb   = static_cast<double>(peak_rss_bytes()) / (1024.0 * 1024.0);
    out.steps     = ad.steps();
    out.n_dummies = ad.num_selected_dummies();
    write_all(out_fd, &out, sizeof(out)); _exit(0);
}

// ----------------------------------------------------
//  run_in_child
// ----------------------------------------------------
template <typename Fn>
static ChildResult run_in_child(Fn fn, int n, int p, int L, int T, uint64_t seed) {
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
//  timestamp
// ----------------------------------------------------
static std::string current_timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}

// ----------------------------------------------------
//  MAIN
// ----------------------------------------------------
int main() {
    const std::string root = "results/sec4_2_memory_runtime";
    const std::string ts   = current_timestamp();

    const std::string run_name = "poisson";

    std::filesystem::path outdir = std::filesystem::path(root) / run_name;
    std::error_code ec;
    std::filesystem::create_directories(outdir, ec);
    if (ec) {
        std::cerr << "Failed to create directory " << outdir << ": " << ec.message() << "\n";
        return 1;
    }

    std::filesystem::path raw_path  = outdir / "raw.csv";
    std::filesystem::path meta_path = outdir / "meta.json";

    std::ofstream raw(raw_path);
    if (!raw) {
        std::cerr << "Failed to open raw.csv for writing at " << raw_path << "\n";
        return 1;
    }

    // Flush after every write: without this the whole CSV sits in the stream
    // buffer until close(), so a job that dies late leaves an empty file.
    raw << std::unitbuf;

    raw << "n,p,m,L,rep,"
        << "time_vd,time_ad,"
        << "mem_vd_mb,mem_ad_mb,"
        << "steps_vd,steps_ad,dummies_vd,dummies_ad,"
        << "ok_vd,ok_ad\n";

    int total = NUM_P * NUM_M * REPS;
    int cnt   = 0;

    for (int pi = 0; pi < NUM_P; ++pi) {
        for (int mi = 0; mi < NUM_M; ++mi) {
            for (int rep = 0; rep < REPS; ++rep) {
                int p = P_LIST[pi];
                int m = L_MULT[mi];
                int L = m * p;
                uint64_t seed = BASE_SEED + 97 * p + 7 * m + 10000 * rep;

                ChildResult vd = run_in_child(child_vd_pois, N_FIXED, p, L, T_SELECT, seed);
                ChildResult ad = run_in_child(child_ad_pois, N_FIXED, p, L, T_SELECT, seed);

                raw
                    << N_FIXED << ',' << p << ',' << m << ',' << L << ',' << rep << ','
                    << vd.sec       << ',' << ad.sec       << ','
                    << vd.peak_mb   << ',' << ad.peak_mb   << ','
                    << vd.steps     << ',' << ad.steps     << ','
                    << vd.n_dummies << ',' << ad.n_dummies << ','
                    << vd.ok        << ',' << ad.ok        << '\n';

                progress(++cnt, total, p, m, L, rep);
            }
        }
    }
    raw.close();
    std::cerr << "\nWrote raw CSV to: " << raw_path << "\n";

    std::ofstream meta(meta_path);
    if (!meta) {
        std::cerr << "Failed to open meta.json for writing at " << meta_path << "\n";
        return 1;
    }

    double vd_cap_gb = env_mem_cap_gb("VD_MEM_CAP_GB");
    double ad_cap_gb = env_mem_cap_gb("ADPOIS_MEM_CAP_GB");

    meta << "{\n";
    meta << "  \"run_name\": \"" << run_name << "\",\n";
    meta << "  \"methods\": [\"vd\", \"ad\"],\n";
    meta << "  \"method_desc\": {\"vd\": \"VD-AFS-Poisson (Eigen)\", "
            "\"ad\": \"AD-AFS-Poisson (Eigen, this paper)\"},\n";
    meta << "  \"family\": \"poisson\",\n";
    meta << "  \"AFS_RHO\": " << AFS_RHO << ",\n";
    meta << "  \"AMP_SD\": " << AMP_SD << ",\n";
    meta << "  \"BASE_RATE\": " << BASE_RATE << ",\n";
    meta << "  \"K_ACTIVE\": " << K_ACTIVE << ",\n";
    meta << "  \"mem_unit\": \"MiB\",\n";
    meta << "  \"timestamp\": \"" << ts << "\",\n";
    meta << "  \"N_FIXED\": " << N_FIXED << ",\n";
    meta << "  \"T_SELECT\": " << T_SELECT << ",\n";
    meta << "  \"REPS\": " << REPS << ",\n";
    meta << "  \"BASE_SEED\": " << BASE_SEED << ",\n";

    meta << "  \"P_LIST\": [";
    for (int i = 0; i < NUM_P; ++i) { meta << P_LIST[i]; if (i + 1 < NUM_P) meta << ", "; }
    meta << "],\n";

    meta << "  \"L_MULT\": [";
    for (int i = 0; i < NUM_M; ++i) { meta << L_MULT[i]; if (i + 1 < NUM_M) meta << ", "; }
    meta << "],\n";

    meta << "  \"VD_MEM_CAP_GB\": ";
    if (std::isinf(vd_cap_gb)) meta << "null"; else meta << vd_cap_gb;
    meta << ",\n";
    meta << "  \"ADPOIS_MEM_CAP_GB\": ";
    if (std::isinf(ad_cap_gb)) meta << "null"; else meta << ad_cap_gb;
    meta << "\n";

    meta << "}\n";
    meta.close();

    std::cerr << "Wrote meta.json to: " << meta_path << "\n";
    return 0;
}
