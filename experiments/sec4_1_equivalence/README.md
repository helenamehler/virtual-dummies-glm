# Sec. 4.1: equivalence and misspecification

Four SNR sweeps, one per response model, each with the four methods of Fig. 1:

| Method | What it is |
|--------|------------|
| AD-GLM | T-Rex with explicitly augmented dummies, correct model |
| VD-GLM | the proposed selector, correct model |
| VD-Gauss | the same selector under the misspecified linear model |
| MX-Knockoff | model-X knockoffs with a lasso statistic in the correct model |

Settings: `n = 300`, `p = 1000`, `s = 10`, `B = 20` random experiments,
`L = 5p`, target FDR 0.1, 500 replicates, OMP (`rho = 1`). Bernoulli at
prevalence 1/2, Poisson at mean count 2, multinomial with `C = 4`, Cox with 40%
censoring and an exponential baseline hazard.

## Running it

```bash
# from the repository root
python experiments/sec4_1_equivalence/sim_snr_poisson.py     # and logistic, cox, multinomial
python experiments/sec4_1_equivalence/plot_snr_panel.py      # Fig. 1
```

Each sweep writes `results/sec4_1_equivalence/<model>/summary.csv` (mean FDP and
TPP with Monte Carlo standard errors per method and SNR) and `meta.json`.
`plot_snr_panel.py` draws `results/sec4_1_equivalence/fig1.pdf` from the four
summaries; `--stem fig4_main` names it like the paper's `\includegraphics`.

The sweeps ran on the Lichtenberg cluster at TU Darmstadt, one job per model on
96 cores, about 8 h each:

```bash
sbatch experiments/sec4_1_equivalence/submit_sweep.sh poisson   # set <project> first
```

Outside Slurm, set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`, otherwise joblib and BLAS each open a thread pool. The
knockoff method needs `knockpy`, `skglm` and, for Cox, `scikit-survival`.

## Files

| File | Role |
|------|------|
| `sim_snr_logistic.py`, `sim_snr_poisson.py`, `sim_snr_cox.py`, `sim_snr_multinomial.py` | the four sweeps |
| `AD_AFS_Logistic.py`, `AD_AFS_Poisson.py`, `AD_AFS_Multinomial.py`, `AD_AFS_Cox.py` | T-Rex with explicitly augmented dummies |
| `snr_glm.py` | the SNR definition per model and the calibration to a target SNR |
| `knockoff_glm.py` | the model-X knockoff method |
| `plot_snr_panel.py` | Fig. 1 |
| `submit_sweep.sh` | Slurm wrapper, one model per job |
