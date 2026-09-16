# Results

The numbers behind the paper, one folder per subsection, like `experiments/`.
Only what a table or figure of the paper needs.

```
results/
├── sec4_1_equivalence/          Sec. 4.1, Fig. 1
│   ├── logistic/                summary.csv, meta.json
│   ├── poisson/                 summary.csv, meta.json
│   ├── cox/                     summary.csv, meta.json
│   ├── multinomial/             summary.csv, meta.json
│   └── fig1.pdf, fig1.png       the figure, drawn from the four summary.csv
├── sec4_2_memory_runtime/       Sec. 4.2
│   └── <model>/                raw.csv, meta.json
├── sec4_3_hapnest/              Sec. 4.3, Table 2
│   └── table2.csv
└── sec4_4_kirc/                 Sec. 4.4, Table 3
    ├── table3.csv
    └── selected_genes.csv
```

| File | Content | Produced by |
|------|---------|-------------|
| `sec4_1_equivalence/<model>/summary.csv` | mean FDP and TPP over 500 replicates with Monte Carlo standard errors (Cox only), one row per method and SNR | `experiments/sec4_1_equivalence/sim_snr_<model>.py` |
| `sec4_1_equivalence/<model>/meta.json` | the settings of that sweep | same |
| `sec4_1_equivalence/fig1.pdf` | Fig. 1 | `experiments/sec4_1_equivalence/plot_snr_panel.py` |
| `sec4_2_memory_runtime/<model>/raw.csv` | one row per replicate: peak RSS in MiB and wall-clock seconds for VD and AD | `bench_memory_<model>` |
| `sec4_2_memory_runtime/<model>/meta.json` | the settings of that benchmark | same |
| `sec4_3_hapnest/table2.csv` | FDP, TPP, selected SNPs and runtime per method, with standard errors over the replicates | `experiments/sec4_3_hapnest/`, condensed by `make_table2.py` |
| `sec4_4_kirc/table3.csv` | selected genes per method, error criterion, and `T`, `L`, `v` for the VD rows | `experiments/sec4_4_kirc/kirc_selection.py`, condensed by `make_table3.py` |
| `sec4_4_kirc/selected_genes.csv` | the genes of the two VD rows with marginal diagnostics | same |

`experiments/sec4_2_memory_runtime/summarize_benchmark.py` turns a `raw.csv`
into the numbers quoted in Sec. 4.2. `plot_snr_panel.py` redraws Fig. 1 from
the summaries in a second.

## Not here

**The knockoff row of Table 3.** A separate cluster job, see
`experiments/sec4_4_kirc/README.md`, step 3.

**Anything large.** Genotypes, expression matrices and `.npz` caches go to
`data/` and are excluded by `.gitignore`. The HAPNEST genotypes are downloaded
from Harvard Dataverse, the TCGA-KIRC data from UCSC Xena.
