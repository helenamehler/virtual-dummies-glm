# Sec. 4.2: memory and runtime

Four C++ drivers, one per family, each forking a child process per run so peak
RSS is measured cleanly. The AD child holds `X` and `D` at once.

Point measured: `n = 10^4`, `p = 10^5`, `L = 5p`, `T = 10`, `rho = 1`, 20
replicates, single-threaded.

## Running it

```bash
# from the repository root, needs Armadillo
cmake -B build -DBUILD_BENCHMARKS=ON
cmake --build build --target bench_memory_poisson -j
./build/bench_memory_poisson          # writes results/sec4_2_memory_runtime/poisson/

python experiments/sec4_2_memory_runtime/summarize_benchmark.py results/sec4_2_memory_runtime/poisson
```

On a cluster, after building:

```bash
sbatch experiments/sec4_2_memory_runtime/submit_benchmark.sh poisson   # set <project> first
```

Targets: `bench_memory_logistic`, `bench_memory_poisson`,
`bench_memory_multinomial`, `bench_memory_cox`.

Set `VD_MEM_CAP_GB` and the family's AD cap (`ADLOGIT_MEM_CAP_GB`,
`ADPOIS_MEM_CAP_GB`, `ADMULTI_MEM_CAP_GB`, `ADCOX_MEM_CAP_GB`) to skip runs
that would exceed the node. The AD side needs about 48 GB at this point, so 70
GB is a sane cap.

## Result

Medians over the 20 replicates, as `summarize_benchmark.py` prints them from the
results in this repository:

| Family | VD overhead | AD overhead | VD runtime | AD runtime | Speedup |
|--------|-------------|-------------|------------|------------|---------|
| Bernoulli | 206 MB | 40.07 GB | 26.6 s | 59.1 s | 2.22x |
| Poisson | 205 MB | 40.07 GB | 24.9 s | 57.9 s | 2.33x |
| Multinomial | 411 MB | 40.09 GB | 27.1 s | 57.5 s | 2.12x |
| Cox | 205 MB | 40.06 GB | 36.5 s | 89.2 s | 2.45x |

Overhead is peak RSS minus the `n*p*8 = 8.0 GB` needed for `(X, y)`. The
multinomial figure is twice the others because the score has `C-1` components,
so the projection block grows by the same factor.

## Files

| File | Role |
|------|------|
| `bench_memory_<family>.cpp` | the four drivers |
| `ad_afs_<family>.hpp` | the AD baselines, C++ ports of `../sec4_1_equivalence/AD_AFS_*.py` |
| `bench_harness.hpp` | fork, CPU pinning and RSS measurement, shared by all four |
| `summarize_benchmark.py` | turns `raw.csv` into the numbers above |
| `submit_benchmark.sh` | Slurm wrapper, one family per job |

Linux and macOS only, since the harness uses `fork()` and `getrusage()`.

The results in `results/sec4_2_memory_runtime/multinomial/` were measured before the
multinomial rank statistic was made reference-invariant. The change adds one
row sum per candidate
and no memory, so the numbers carry over, but a rerun is the clean way to
confirm it.
