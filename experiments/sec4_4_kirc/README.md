# Sec. 4.4: survival in clear-cell renal cell carcinoma

Table 3. Gene expression and overall survival from TCGA-KIRC: `n = 531`
patients, `p = 19 620` protein-coding genes, 175 deaths. Target FDR 0.1,
`B = 20`, `L = 10p`, adaptive forward stepwise with `rho = 0.1`, `T_max = 50`.

The Table 3 caption of the draft says `L = 5p`; the run behind the table used
`L = 10p` (`L = 196 200` in `table3.csv`).

## Running it

From the repository root; the defaults of both scripts are the paper setting.

```bash
# 1. download and preprocess TCGA-KIRC into data/kirc_OS.npz
python experiments/sec4_4_kirc/kirc_data.py

# 2. all rows of Table 3 except knockoffs -> results/sec4_4_kirc/rerun/
python experiments/sec4_4_kirc/kirc_selection.py --threads 8

# 3. the knockoff row; the construction is cached, it is the expensive half
python experiments/sec4_4_kirc/cox_knockoff.py generate data/kirc_OS.npz results/sec4_4_kirc/rerun/knockoff/ko_cache \
    '{"cohort":"KIRC","ko_method":"equicorrelated","ko_shrinkage":"ledoitwolf","seed":123}'
python experiments/sec4_4_kirc/kirc_selection.py --methods cox_knockoff \
    --ko-method equicorrelated --ko-cache-dir results/sec4_4_kirc/rerun/knockoff/ko_cache \
    --out-dir results/sec4_4_kirc/rerun/knockoff

# 4. table3.csv and selected_genes.csv
python experiments/sec4_4_kirc/make_table3.py results/sec4_4_kirc/rerun results/sec4_4_kirc/rerun/knockoff \
    --out results/sec4_4_kirc/rerun
```

`kirc_data.py` downloads the gene-level STAR counts (UCSC Xena, GDC hub),
overall survival from the Pan-Cancer Atlas clinical data resource and the
protein-coding gene list of the Human Protein Atlas, all without login. It keeps
primary tumors and protein-coding genes, drops genes with a total count below 10
and patients with zero follow-up, applies median-of-ratios size factors and
`log2(K/s + 1)`, and centers and unit-normalizes every gene. It warns if the
result is not `n = 531`, `p = 19 620`, 175 events; a newer Human Protein Atlas
release can move `p` by a few genes.

On a Mac with NumPy 2, Apple's Accelerate BLAS makes NumPy report spurious
`RuntimeWarning: ... encountered in matmul` in the Cox Lasso. `kirc_selection.py`
silences exactly that message on macOS.

**Platform.** The dummies are drawn with the random distributions of the C++
standard library, which differ between macOS (libc++) and Linux (libstdc++).
`results/sec4_4_kirc/` was computed on macOS. On Linux, VD-AFS-Cox selects nine
genes (the eight of the paper and CCL7) and the Gaussian row none; the other
rows are identical.

## Files

| File | Role |
|------|------|
| `kirc_data.py` | download and preprocessing, writes `data/kirc_OS.npz` |
| `kirc_selection.py` | VD-AFS-Cox and the benchmark rows of Table 3 |
| `cox_knockoff.py` | the model-X knockoff row |
| `make_table3.py` | condenses a run into `table3.csv` and `selected_genes.csv` |

## What's in `results/sec4_4_kirc/`

| File | Content |
|------|---------|
| `table3.csv` | selected genes per method, error criterion, and `T`, `L`, `v` for the VD rows |
| `selected_genes.csv` | the 8 genes of VD-AFS-Cox and the 1 of the Gaussian model on `log t`, with marginal Cox z-score and correlation with `log t` and the event indicator |

The knockoff row of Table 3 (0 genes) came from a separate cluster job whose
output is not in this folder; step 3 rebuilds it.
