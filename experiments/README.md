# Experiments

Everything behind the numbers in the paper, one folder per subsection of
Sec. 4.

| Folder | Paper | What it is |
|--------|-------|------------|
| [`sec4_1_equivalence/`](sec4_1_equivalence/README.md) | Sec. 4.1, Fig. 1 | equivalence of AD and VD, misspecification, knockoffs |
| [`sec4_2_memory_runtime/`](sec4_2_memory_runtime/README.md) | Sec. 4.2 | peak memory and runtime at `n = 10^4`, `p = 10^5` |
| [`sec4_3_hapnest/`](sec4_3_hapnest/README.md) | Sec. 4.3, Table 2 | disease subtype classification on HAPNEST genotypes |
| [`sec4_4_kirc/`](sec4_4_kirc/README.md) | Sec. 4.4, Table 3 | survival in clear-cell renal carcinoma, TCGA-KIRC |

The results the paper reports are already in `../results/`, so the plotting
and summary scripts run in seconds. Regenerating the results is the expensive
part:

| Paper | Output | Script | Cost |
|-------|--------|--------|------|
| Sec. 4.1, Fig. 1 | `results/sec4_1_equivalence/<model>/` | `sec4_1_equivalence/sim_snr_<model>.py` | ~8 h per model on 96 cores |
| Sec. 4.1, Fig. 1 | `results/sec4_1_equivalence/fig1.pdf` | `sec4_1_equivalence/plot_snr_panel.py` | seconds |
| Sec. 4.2 | `results/sec4_2_memory_runtime/<model>/` | `sec4_2_memory_runtime/bench_memory_<model>.cpp` | ~1 h per model, 70 GB RAM |
| Sec. 4.2 | the numbers in the text | `sec4_2_memory_runtime/summarize_benchmark.py` | seconds |
| Sec. 4.3, Table 2 | `results/sec4_3_hapnest/table2.csv` | `sec4_3_hapnest/` | ~7 min per replicate for VD-Multinomial, ~1 h plus construction for knockoffs |
| Sec. 4.4, Table 3 | `results/sec4_4_kirc/table3.csv` | `sec4_4_kirc/` | minutes, knockoffs hours |

## Setup

```bash
# from the repository root
pip install -e .
pip install -r experiments/requirements.txt
```

The HAPNEST application has its own
[`requirements.txt`](sec4_3_hapnest/requirements.txt). Run every script from the
repository root; the scripts write into `results/` relative to it.

`AFS` with `rho = 1` is OMP, which the paper uses in Sec. 4.1 to 4.3. The KIRC
application uses `rho = 0.1`, adaptive forward stepwise.
