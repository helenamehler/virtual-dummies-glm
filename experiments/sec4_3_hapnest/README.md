# Sec. 4.3: disease subtype classification

Table 2. `n = 10 000` individuals, 30 987 to 31 139 SNPs on chromosome 1
depending on the replicate, `C = 4` classes with prevalences
`(1/2, 1/6, 1/6, 1/6)`, `s = 12` causal SNPs of which 4 act on all subtypes,
target FDR 0.1, `B = 20`, `L = 5p`.

Labels are drawn from a softmax under a multiplicative relative-risk model.
Shared SNPs have `RR ~ Unif(1.15, 1.30)` on every subtype (with a per-subtype
jitter of +-15%), subtype-specific SNPs `RR ~ Unif(1.20, 1.42)` on one subtype.
The direction of each effect, risk or protective, is drawn at random.

```
1_genotype_generation/   HAPNEST config and Slurm scripts (Koka et al., only needed to simulate from scratch)
2_preprocessing/         download, LD pruning, standardization
3_4_benchmark/           phenotype, the Table 2 methods, knockoffs
```

### Getting the genotypes

The genotypes are the small-scale HAPNEST cohort of Koka et al. (2026), which
Taulant Koka deposited on Harvard Dataverse. Nothing needs to be simulated and
no login is needed.

| Set | DOI | Content |
|-----|-----|---------|
| small-scale | [10.7910/DVN/RZ3FZT](https://doi.org/10.7910/DVN/RZ3FZT) | 100 replicates, chromosome 1, `n = 10 000`, ~94 500 SNPs each, ~24 GB |

All sets are listed at <https://dataverse.harvard.edu/dataverse/taulantkoka>.
The files are already quality controlled with PLINK (MAF > 1%, call rate >
95%, HWE p > 1e-6). Please cite Koka et al. (2026) and HAPNEST (Wharrie et al.,
2023) when you use them.

### From download to `X_std.dat`

One script does it, with the same preprocessing as Koka et al.:

```bash
cd experiments/sec4_3_hapnest
pip install -r requirements.txt

# a pilot on three replicates first
RUNS=1-3 bash prepare_hapnest_small.sh all

# then everything
bash prepare_hapnest_small.sh all
```

`all` runs four steps, which can also be called one at a time:

| Step | What it does |
|------|--------------|
| `download` | `2_preprocessing/download_hapnest_small.py`, one file at a time, restartable, size-checked |
| `arrange` | `2_preprocessing/arrange_hapnest_runs.py`. Dataverse delivers the files flat (`run_17_chr-1.bed`), the preprocessing reads `run_17/run_17_chr-1.bed`. Symlinks, no copies |
| `preprocess` | `2_preprocessing/preprocess_hapnest_small.py` per replicate: LD pruning by single-linkage clustering at `|corr| > 0.7` (correlations on a random subsample of 3000 rows), one random representative per cluster, columns centered and unit-L2 normalized |
| `check` | prints `n`, `p` and the maximum remaining `|corr|` per replicate |

Paths are set by environment variables: `RAW_DIR` for the download (default
`data/hapnest_raw_small` in the repository root), `DATA_DIR` for the output
(default `data/hapnest_preprocessed_small`), `RUNS` for the replicates.

Pruning leaves about 31 000 SNPs per replicate, with residual LD between them.
Each replicate ends up as

```
run_<id>/
├── X_std.dat       # centered, unit-L2 design matrix (n x p, F-order, float64)
├── X_raw.dat       # raw 0/1/2 dosages (n x p, F-order, float64)
├── meta.json       # n_samples, p_pruned_total, seed, max |corr| after pruning
└── kept_snps.txt   # IDs of the retained SNPs
```

and `DATA_DIR` is what the benchmark reads.

**Budget.** Every preprocessed replicate is two dense `10 000 x 31 000`
float64 matrices, about 5 GB, so all 100 need about 500 GB of disk. Put
`DATA_DIR` on scratch. Preprocessing one replicate takes about 24 GB of RAM.

**On a cluster** the 100 replicates are a job array. Download and arrange on a
login or transfer node, then

```bash
mkdir -p logs
sbatch submit_preprocess_small.sh        # set <project>, RAW_DIR, DATA_DIR first
bash prepare_hapnest_small.sh check
```

`submit_preprocess_small.sh` is the Slurm wrapper of the upstream pipeline with
the paths made configurable.

**Simulating from scratch** instead of downloading needs
[HAPNEST](https://github.com/simlab-bioinf/hapnest) (Julia), the 1000 Genomes +
HGDP reference panels from the HAPNEST documentation, PLINK 1.9 and PLINK 2.
`1_genotype_generation/` holds Koka's three files, unchanged:

| File | What it does |
|------|--------------|
| `config_100k_template.yaml` | HAPNEST config with placeholders `__SEED__`, `__CHR__`, `__RUNID__`, `__OUTDIR__`, `__DATADIR__` |
| `slurm_geno_chr.sh` | one array task per chromosome: fills the placeholders, runs `run_program.jl --genotype`, then PLINK QC (MAF > 1%, call rate > 95%, HWE p > 1e-6) |
| `submit_all_runs.sh` | submits `slurm_geno_chr.sh` once per replicate, passing `RUN_ID` |

They are set up for the full-scale cohort of Koka et al. and for their
cluster account, so for the small-scale cohort used here change

- `nsamples: 100000` to `10000` in the config,
- `N_RUNS=30` to `100` in `submit_all_runs.sh`,
- `#SBATCH --array=1-22` to `1` in `slurm_geno_chr.sh` (chromosome 1 only),
- the account (`-A`), `SCRATCH_DIR`, `HAPNEST_DIR`, `DATA_DIR`, the module
  names and the tool paths under `software:` in the config.

`submit_all_runs.sh` expects the config and `slurm_geno_chr.sh` in
`~/hapnest/`, so copy them there or adjust the two paths. The output lands in
`$SCRATCH_DIR/outputs/run_<id>/`, which is already the layout the preprocessing
reads, so `prepare_hapnest_small.sh preprocess` works on it directly with
`RAW_DIR=$SCRATCH_DIR/outputs`. The simulation is random, so the genotypes
will not be bit-identical to the Dataverse deposit.

### Phenotypes and methods

```bash
cd experiments/sec4_3_hapnest
# DATA_DIR defaults to data/hapnest_preprocessed_small in the repository root

bash run_multinomial_benchmark.sh pilot          # 3 replicates
N_REPS=100 bash run_multinomial_benchmark.sh full
N_REPS=50 bash run_knockoff_rows.sh full         # the knockoff row, separately
bash run_multinomial_benchmark.sh table          # table2.csv
```

Everything lands under `results/sec4_3_hapnest/rerun/`: phenotypes, one JSON
per replicate and method, and `table2.csv`. The two drivers use the same
phenotypes, so the knockoff row is computed on the same data as the rest.

| Key | Table 2 row |
|-----|-------------|
| `trex_multinom` | VD-Multinomial |
| `trex_logit_ovr` | VD-Bernoulli, OvR, each class at `alpha/(C-1)` |
| `trex_gauss` | VD-Gaussian on the centered integer labels |
| `ko_mx_multinom` | Model-X Knockoffs, multinomial lasso statistic |
| `bh_multinom`, `by_multinom` | Multinomial Score + BH / BY |

Labels must stay raw integers `0..C-1`; the multinomial solver reads them
directly. `max_vd_proj` is raised to 600 (300 for the Bernoulli rows) because
the multinomial score reveals up to `C-1` directions per step.

One replicate of VD-Multinomial takes about 6.5 minutes on one thread. The
knockoff construction is cached per replicate; the statistic alone takes about
1 hour.

## What's in `results/sec4_3_hapnest/`

`table2.csv`, one row per Table 2 method: replicates, mean FDP, TPP and number
of selected SNPs with standard errors, and mean runtime per replicate in
minutes. It was built with `make_table2.py` from the per-replicate JSONs of the
Lichtenberg run:

- The T-Rex and marginal rows come from the run of 14 Sep 2026, after the
  multinomial rank statistic was made reference-invariant.
- The knockoff row comes from the run of 10 Sep 2026 on the same phenotypes.
  Knockoffs do not use the rank statistic.
- Replicate 28 is missing from all rows, its preprocessing did not finish.
  Replicate 79 is missing from the Bernoulli one-vs-rest row, which stopped
  with a division by zero.
- The knockoff runtime covers only the statistic, not the cached construction.

## Dependencies

```bash
pip install -r requirements.txt
```
