// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// experiments/sec4_2_memory_runtime/bench_memory_cox.cpp
//
// 2-way runtime/memory benchmark for the COX AFS family:
//   * VD-AFS-Cox  (Eigen, src/vd_afs_cox.hpp)          -> columns *_vd
//   * AD-AFS-Cox  (Eigen, experiments/sec4_2_memory_runtime/ad_afs_cox.hpp)  -> columns *_ad
//
// Fabian's TRexSelector covers the linear model only, so the explicit-dummy
// baseline is our own C++ port of experiments/sec4_1_equivalence/AD_AFS_Cox.py. Both sides are
// Eigen, single-threaded, and run in separate forked processes on identical
// generated data. The harness lives in bench_harness.hpp.
//
// CLUSTER CONFIG (ICASSP memory point):
//   n = 10 000, p = 100 000, L = 5p = 500 000, T = 10, rho = 1.0, 20 replicates.
//   AD child: X and D at once, n*(p+L)*8 = 4.8e10 B = 44.7 GiB.
//   VD child: X only, n*p*8 = 8.0e9 B = 7.45 GiB, plus the k x L projections.
//   Request at least 64 GB and keep ADCOX_MEM_CAP_GB above 58 (the code
//   multiplies the requirement by a 1.3 safety factor before comparing).
//
// Walltime note: the Cox refit is the most expensive of the four families per
// Newton iteration -- every iteration sweeps all n samples in descending time
// and accumulates a k x k outer product per sample, so the refit is O(n k^2)
// where the GLM families are O(n k). The correlation refresh, which is what VD
// actually saves, is identical to the other families.
//
// Both columns are timed: `time_vd` / `time_ad` cover run() only -- data
// generation, the dummy block and the memory-cap check are outside the clock.
//
// Memory columns are MiB (bytes / 1024^2); meta.json records the unit.
// For the paper, convert with MiB * 1.048576 = MB.
#include "bench_harness.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

#include "vd_afs_cox.hpp"        // VD side (existing; found via -I src)
#include "ad_afs_cox.hpp"        // AD side (this paper)

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
static const double AMP       = 1.0;   // sd of the linear predictor
static const double CENS_RATE = 0.3;   // target right-censoring rate
static const double AFS_RHO   = 1.0;   // 1.0 = GLM-OMP (both solvers)
static const bool   USE_EFRON = false; // Breslow ties, both sides

// ----------------------------------------------------
//  CENSORING SCALE
// ----------------------------------------------------
// Exponential censoring scale c with E[censoring rate] == target. For
// C ~ Exp(scale = c) independent of Ttrue, P(C < T_i) = 1 - exp(-T_i / c), so
// the expected rate is mean_i(1 - exp(-T_i/c)) -- strictly decreasing in c, so
// bisection hits the target exactly. Mirrors _solve_cens_scale in
// experiments/sec4_1_equivalence/sim_snr_cox.py; the quantile heuristic it
// replaced misses the target by a different amount at each amplitude.
static double solve_cens_scale(const arma::vec& Ttrue, double target) {
    double lo = 1e-8;
    double hi = std::max(Ttrue.max() * 1e4, 1.0);
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double rate = arma::mean(1.0 - arma::exp(-Ttrue / mid));
        if (rate > target) lo = mid; else hi = mid;
        if (hi - lo < 1e-10 * std::max(1.0, hi)) break;
    }
    return 0.5 * (lo + hi);
}

// ----------------------------------------------------
//  DATA GEN -- survival times + censoring, shared by both children
// ----------------------------------------------------
//
// Mirrors make_problem / sample_y_amp in
// experiments/sec4_1_equivalence/sim_snr_cox.py:
//   beta_base has K_ACTIVE entries of +-1
//   eta = AMP * (X beta) / sd(X beta)          -> Var(eta) ~ AMP^2
//   Ttrue = -log(U) / exp(eta)                 -> inverse-CDF exponential
//   C ~ Exp(scale calibrated to CENS_RATE), time = min(Ttrue, C),
//   delta = 1[Ttrue <= C]
// The response y carries the TIMES; delta goes to the solvers separately.
static void gen_problem_cox(int n, int p, uint64_t seed,
                            arma::mat& X, arma::vec& y, arma::vec& delta)
{
    arma::arma_rng::set_seed(seed);
    X = arma::randn(n, p);
    center_unit_l2(X);

    std::mt19937_64 rng(seed + 7);
    std::uniform_int_distribution<int> coin(0, 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    arma::vec beta_base(p, arma::fill::zeros);
    const int s = std::min(K_ACTIVE, p);
    for (int j = 0; j < s; ++j) beta_base(j) = coin(rng) ? 1.0 : -1.0;

    arma::vec signal = X * beta_base;
    const double sd = std::sqrt(std::max(arma::var(signal), 1e-20));
    signal /= sd;
    arma::vec eta = AMP * signal;

    arma::vec Ttrue(n);
    for (int i = 0; i < n; ++i) {
        const double u = std::max(unif(rng), 1e-300);
        Ttrue(i) = -std::log(u) / std::exp(eta(i));
    }

    y.set_size(n);
    delta.set_size(n);
    if (CENS_RATE <= 0.0) {
        y = Ttrue;
        delta.ones();
    } else {
        const double scale = solve_cens_scale(Ttrue, CENS_RATE);
        for (int i = 0; i < n; ++i) {
            const double u = std::max(unif(rng), 1e-300);
            const double c = -scale * std::log(u);          // Exp(scale)
            y(i)     = std::min(Ttrue(i), c);
            delta(i) = (Ttrue(i) <= c) ? 1.0 : 0.0;
        }
    }
    // NOTE: no centering. y carries raw event/censoring times.
}

// ----------------------------------------------------
//  CHILD WORKERS
// ----------------------------------------------------
static void child_vd_cox(int n, int p, int L, int T,
                         uint64_t seed, int out_fd) {
    pin_to_core(2);
    cap_blas_env();
    if (refuse_if_over_cap(vd_bytes_needed(n, p), "VD_MEM_CAP_GB", 1.0L, out_fd))
        _exit(0);

    ChildResult out{}; out.ok = 1; out.code = 0;

    arma::mat X; arma::vec y, delta;
    gen_problem_cox(n, p, seed, X, y, delta);

    using MatCol = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    Eigen::Map<const MatCol>          Xeig(X.memptr(), n, p);
    Eigen::Map<const Eigen::VectorXd> Yeig(y.memptr(), n);

    VDOptions opt;
    opt.T_stop      = T;
    opt.max_vd_proj = n;
    opt.standardize = false;
    opt.rho         = AFS_RHO;
    opt.seed        = seed;
    opt.cox_efron   = USE_EFRON;
    opt.cox_status  = Eigen::Map<const Eigen::VectorXd>(delta.memptr(), n);

    VD_AFS_Cox vd(Xeig.data(), n, p, Yeig.data(), n, L, opt);

    auto t0 = std::chrono::high_resolution_clock::now();
    (void)vd.run(T);
    auto t1 = std::chrono::high_resolution_clock::now();

    out.sec       = std::chrono::duration<double>(t1 - t0).count();
    out.peak_mb   = peak_rss_mib();
    // Path length, i.e. one entry per selection step (real or dummy). NOT
    // basis_size(): that counts the orthogonal projection basis, which also
    // holds the score direction and one row per revealed dummy direction, so
    // it runs well above the step count and is not comparable to AD's step_.
    out.steps     = static_cast<int>(vd.active_features_copy().size());
    out.n_dummies = vd.num_realized_dummies();
    write_all(out_fd, &out, sizeof(out)); _exit(0);
}

static void child_ad_cox(int n, int p, int L, int T,
                         uint64_t seed, int out_fd) {
    pin_to_core(2);
    cap_blas_env();
    if (refuse_if_over_cap(ad_bytes_needed(n, p, L), "ADCOX_MEM_CAP_GB", 1.3L, out_fd))
        _exit(0);

    ChildResult out{}; out.ok = 1; out.code = 0;

    arma::mat X; arma::vec y, delta;
    gen_problem_cox(n, p, seed, X, y, delta);

    arma::mat D;
    gen_dummies(n, L, seed, D);

    AD_AFS_Cox::Options o;
    o.rho   = AFS_RHO;
    o.efron = USE_EFRON;

    // X and D stay separate: concatenating would double the peak RSS and the
    // benchmark would measure the copy instead of the algorithm.
    AD_AFS_Cox ad(X.memptr(), n, p, D.memptr(), L,
                  y.memptr(), delta.memptr(), o);

    auto t0 = std::chrono::high_resolution_clock::now();
    (void)ad.run(T);
    auto t1 = std::chrono::high_resolution_clock::now();

    out.sec       = std::chrono::duration<double>(t1 - t0).count();
    out.peak_mb   = peak_rss_mib();
    out.steps     = ad.steps();
    out.n_dummies = ad.num_selected_dummies();
    write_all(out_fd, &out, sizeof(out)); _exit(0);
}

// ----------------------------------------------------
//  MAIN
// ----------------------------------------------------
int main() {
    const std::string root = "results/sec4_2_memory_runtime";
    const std::string ts   = current_timestamp();

    const std::string run_name = "cox";

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

                ChildResult vd = run_in_child(child_vd_cox, N_FIXED, p, L, T_SELECT, seed);
                ChildResult ad = run_in_child(child_ad_cox, N_FIXED, p, L, T_SELECT, seed);

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
    double ad_cap_gb = env_mem_cap_gb("ADCOX_MEM_CAP_GB");

    meta << "{\n";
    meta << "  \"run_name\": \"" << run_name << "\",\n";
    meta << "  \"methods\": [\"vd\", \"ad\"],\n";
    meta << "  \"method_desc\": {\"vd\": \"VD-AFS-Cox (Eigen)\", "
            "\"ad\": \"AD-AFS-Cox (Eigen, this paper)\"},\n";
    meta << "  \"family\": \"cox\",\n";
    meta << "  \"ties\": \"" << (USE_EFRON ? "efron" : "breslow") << "\",\n";
    meta << "  \"AFS_RHO\": " << AFS_RHO << ",\n";
    meta << "  \"AMP\": " << AMP << ",\n";
    meta << "  \"CENS_RATE\": " << CENS_RATE << ",\n";
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
    meta << "  \"ADCOX_MEM_CAP_GB\": ";
    if (std::isinf(ad_cap_gb)) meta << "null"; else meta << ad_cap_gb;
    meta << "\n";

    meta << "}\n";
    meta.close();

    std::cerr << "Wrote meta.json to: " << meta_path << "\n";
    return 0;
}
