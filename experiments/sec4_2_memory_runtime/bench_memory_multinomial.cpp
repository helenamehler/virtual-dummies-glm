// NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
// experiments/sec4_2_memory_runtime/bench_memory_multinomial.cpp
//
// 2-way runtime/memory benchmark for the MULTINOMIAL AFS family:
//   * VD-AFS-Multinomial  (Eigen, src/vd_afs_multinomial.hpp)         -> *_vd
//   * AD-AFS-Multinomial  (Eigen, experiments/sec4_2_memory_runtime/ad_afs_multinomial.hpp) -> *_ad
//
// Fabian's TRexSelector covers the linear model only, so the explicit-dummy
// baseline is our own C++ port of experiments/sec4_1_equivalence/AD_AFS_Multinomial.py. Both sides
// are Eigen, single-threaded, and run in separate forked processes on identical
// generated data. The harness lives in bench_harness.hpp.
//
// CLUSTER CONFIG (ICASSP memory point):
//   n = 10 000, p = 100 000, L = 5p = 500 000, T = 10, rho = 1.0, C = 4 classes,
//   20 replicates.
//   AD child: X and D at once, n*(p+L)*8 = 4.8e10 B = 44.7 GiB.
//   VD child: X only, n*p*8 = 8.0e9 B = 7.45 GiB, plus the projections.
//
// One difference from the Gaussian / Poisson / logistic runs worth knowing when
// you read the VD memory column: VD_AFS_Multinomial appends up to Cm = C-1
// score directions to the revealed subspace PER STEP (see the header comment in
// src/vd_afs_multinomial.hpp), so the k x L projection block grows roughly Cm
// times faster than in the single-score families. At C = 4 and ~35 steps that
// is ~100 rows x L x 8 B = ~0.4 GiB, still small next to X, but it is the
// reason the VD overhead here is larger than in the Poisson run.
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

#include "vd_afs_multinomial.hpp"     // VD side (existing; found via -I src)
#include "ad_afs_multinomial.hpp"             // AD side (this paper)

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

static const int    N_CLASSES = 4;     // C; classes are 0..C-1, 0 is the reference
static const int    K_ACTIVE  = 10;    // active predictors
static const double AMP       = 1.0;   // log-odds amplitude
static const double AFS_RHO   = 1.0;   // 1.0 = GLM-OMP (both solvers)

static const double ETA_CLAMP = 30.0;

// ----------------------------------------------------
//  DATA GEN -- integer class labels, shared by both children
// ----------------------------------------------------
//
// Mirrors make_problem / sample_y_amp in
// experiments/sec4_1_equivalence/sim_snr_multinomial.py:
//   W_base is p x (C-1); the k-th active predictor loads on class k mod (C-1)
//   Z = X * sqrt(n)          (unit-variance columns)
//   eta = AMP * (Z W_base)   (n x C-1 log-odds against the reference class)
//   Pr = softmax over [0, eta], y ~ Categorical(Pr)
// The labels stay RAW integers 0..C-1: both solvers infer C as max(y)+1 and
// build the class indicators themselves. Do NOT center y here.
static void gen_problem_multinomial(int n, int p, uint64_t seed,
                                    arma::mat& X, arma::vec& y)
{
    const int Cm = N_CLASSES - 1;

    arma::arma_rng::set_seed(seed);
    X = arma::randn(n, p);
    center_unit_l2(X);

    std::mt19937_64 rng(seed + 7);
    std::uniform_int_distribution<int> coin(0, 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    arma::mat W(p, Cm, arma::fill::zeros);
    const int s = std::min(K_ACTIVE, p);
    for (int k = 0; k < s; ++k) W(k, k % Cm) = coin(rng) ? 1.0 : -1.0;

    // Z = X * sqrt(n) restores unit-variance columns, so AMP means the same
    // thing at every n. Fold the sqrt(n) into the coefficient instead of
    // materializing a second n x p block.
    arma::mat eta = X * (W * (AMP * std::sqrt(static_cast<double>(n))));

    y.set_size(n);
    std::vector<double> e(N_CLASSES);
    for (int i = 0; i < n; ++i) {
        e[0] = 1.0;                       // reference class, eta == 0
        double tot = 1.0;
        for (int c = 0; c < Cm; ++c) {
            const double v = std::min(std::max(eta(i, c), -ETA_CLAMP), ETA_CLAMP);
            e[c + 1] = std::exp(v);
            tot += e[c + 1];
        }
        const double u = unif(rng) * tot;
        double acc = 0.0;
        int lab = N_CLASSES - 1;
        for (int c = 0; c < N_CLASSES; ++c) {
            acc += e[c];
            if (u <= acc) { lab = c; break; }
        }
        y(i) = static_cast<double>(lab);
    }
    // NOTE: no centering. The multinomial solvers consume RAW labels.
}

// ----------------------------------------------------
//  CHILD WORKERS
// ----------------------------------------------------
static void child_vd_multi(int n, int p, int L, int T,
                           uint64_t seed, int out_fd) {
    pin_to_core(2);
    cap_blas_env();
    if (refuse_if_over_cap(vd_bytes_needed(n, p), "VD_MEM_CAP_GB", 1.0L, out_fd))
        _exit(0);

    ChildResult out{}; out.ok = 1; out.code = 0;

    arma::mat X; arma::vec y;
    gen_problem_multinomial(n, p, seed, X, y);

    using MatCol = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    Eigen::Map<const MatCol>          Xeig(X.memptr(), n, p);
    Eigen::Map<const Eigen::VectorXd> Yeig(y.memptr(), n);

    VDOptions opt;
    opt.T_stop      = T;
    opt.max_vd_proj = n;
    opt.standardize = false;
    opt.rho         = AFS_RHO;
    opt.seed        = seed;

    VD_AFS_Multinomial vd(Xeig.data(), n, p, Yeig.data(), n, L, opt);

    auto t0 = std::chrono::high_resolution_clock::now();
    (void)vd.run(T);
    auto t1 = std::chrono::high_resolution_clock::now();

    out.sec       = std::chrono::duration<double>(t1 - t0).count();
    out.peak_mb   = peak_rss_mib();
    // Path length, i.e. one entry per selection step (real or dummy). NOT
    // basis_size(): for the multinomial selector the basis grows by up to Cm
    // axes per step, so it is even further from the step count than in the
    // single-score families and not comparable to AD's step_.
    out.steps     = static_cast<int>(vd.active_features_copy().size());
    out.n_dummies = vd.num_realized_dummies();
    write_all(out_fd, &out, sizeof(out)); _exit(0);
}

static void child_ad_multi(int n, int p, int L, int T,
                           uint64_t seed, int out_fd) {
    pin_to_core(2);
    cap_blas_env();
    if (refuse_if_over_cap(ad_bytes_needed(n, p, L), "ADMULTI_MEM_CAP_GB", 1.3L, out_fd))
        _exit(0);

    ChildResult out{}; out.ok = 1; out.code = 0;

    arma::mat X; arma::vec y;
    gen_problem_multinomial(n, p, seed, X, y);

    arma::mat D;
    gen_dummies(n, L, seed, D);

    AD_AFS_Multinomial::Options o;
    o.rho = AFS_RHO;

    // X and D stay separate: concatenating would double the peak RSS and the
    // benchmark would measure the copy instead of the algorithm.
    AD_AFS_Multinomial ad(X.memptr(), n, p, D.memptr(), L, y.memptr(), o);

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

    const std::string run_name = "multinomial";

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

                ChildResult vd = run_in_child(child_vd_multi, N_FIXED, p, L, T_SELECT, seed);
                ChildResult ad = run_in_child(child_ad_multi, N_FIXED, p, L, T_SELECT, seed);

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
    double ad_cap_gb = env_mem_cap_gb("ADMULTI_MEM_CAP_GB");

    meta << "{\n";
    meta << "  \"run_name\": \"" << run_name << "\",\n";
    meta << "  \"methods\": [\"vd\", \"ad\"],\n";
    meta << "  \"method_desc\": {\"vd\": \"VD-AFS-Multinomial (Eigen)\", "
            "\"ad\": \"AD-AFS-Multinomial (Eigen, this paper)\"},\n";
    meta << "  \"family\": \"multinomial\",\n";
    meta << "  \"N_CLASSES\": " << N_CLASSES << ",\n";
    meta << "  \"AFS_RHO\": " << AFS_RHO << ",\n";
    meta << "  \"AMP\": " << AMP << ",\n";
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
    meta << "  \"ADMULTI_MEM_CAP_GB\": ";
    if (std::isinf(ad_cap_gb)) meta << "null"; else meta << ad_cap_gb;
    meta << "\n";

    meta << "}\n";
    meta.close();

    std::cerr << "Wrote meta.json to: " << meta_path << "\n";
    return 0;
}
