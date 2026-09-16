# Virtual dummies for GLMs and Cox regression

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

FDR-controlled variable selection for binary, count, categorical and censored
responses, without storing a dummy matrix.

The T-Rex selector controls the false discovery rate by letting synthetic null
variables compete with the real predictors. The virtual dummy construction of
[Koka et al. (2026)](https://github.com/taulantkoka/virtual-dummies) removes the
memory cost of those dummies by sampling only their projections onto the
subspace the selection has revealed so far. Both were specified for the linear
model. This repository extends the construction to score-based forward selection
and instantiates it for Bernoulli, Poisson, multinomial and Cox responses.

The selection path stays equal in distribution to explicit augmentation, so the
FDR guarantee carries over under the same assumptions. At `n = 10^4`,
`p = 10^5` and `L = 5p`, explicit augmentation adds 40.1 GB of peak memory on
top of the 8.0 GB for `(X, y)`. The virtual version adds about 205 MB and runs
2.2 to 2.5 times faster.

The C++ core, the build system and the
HAPNEST genotype pipeline come from `taulantkoka/virtual-dummies` and are
redistributed under its license, GPLv3. Every file says where it came from in
its first line, and [NOTICE](NOTICE) lists the provenance file by file.

## Installation

```bash
git clone https://github.com/<your-user>/virtual-dummies-glm.git
cd virtual-dummies-glm
pip install .
```

This builds the `vd_selectors` C++ extension via scikit-build-core and
pybind11. Eigen headers are vendored in `extern/`. A system BLAS (OpenBLAS,
MKL, or macOS Accelerate) is used when it's available.

### macOS with Xcode 17+

If the import fails with a `__hash_memory` symbol error:

```bash
export CC=/usr/bin/clang CXX=/usr/bin/clang++
pip install .
```

### Development install

```bash
pip install -e .
```

Check that the GLM solvers made it into the build:

```bash
python -c "from vd_selectors import SolverType; \
print([s for s in dir(SolverType) if not s.startswith('_')])"
# must contain AFS_Poisson, AFS_Multinomial and AFS_Cox
```

## Quick start

### FDR-controlled selection on a count response

```python
import numpy as np
from vd_selectors import TRexSelector, TRexOptions, SolverType, CalibMode, VDDummyLaw

rng = np.random.default_rng(42)
n, p, s = 300, 1000, 10
X = rng.standard_normal((n, p))
X -= X.mean(axis=0); X /= np.linalg.norm(X, axis=0, keepdims=True)
X = np.asfortranarray(X)

beta = np.zeros(p)
beta[rng.choice(p, s, replace=False)] = rng.choice([-1.0, 1.0], s) * 0.35
y = rng.poisson(np.exp(X @ beta)).astype(float)   # raw counts, not centered

opt = TRexOptions()
opt.tFDR = 0.1
opt.K = 20                       # random experiments (B in the paper)
opt.L_factor = 5                 # L = 5p virtual dummies per experiment
opt.solver = SolverType.AFS_Poisson
opt.calib = CalibMode.CalibrateT
opt.rho = 1.0                    # 1.0 = OMP, < 1.0 = adaptive forward stepwise
opt.dummy_law = VDDummyLaw.Spherical
opt.seed = 42

result = TRexSelector(opt).run(X, y)
print("selected:", np.array(result.selected_var))
print(f"T* = {result.T_stop}, v* = {result.v_thresh:.3f}")
```

### Survival response

The response carries the event or censoring times. The event indicator goes in
separately, so no imputation of the censored times is needed.

```python
opt = TRexOptions()
opt.tFDR = 0.1
opt.solver = SolverType.AFS_Cox
opt.cox_status = delta.astype(float)   # 1 = event observed, 0 = right censored
opt.cox_efron = False                  # False = Breslow ties
opt.rho = 0.1                          # small steps help when p >> n
result = TRexSelector(opt).run(X, time)
```

### Multi-class response

Labels stay raw integers `0..C-1`, with class 0 as the reference. The number of
classes is read off the labels. One caveat: the revealed subspace grows by up to
`C-1` axes per step instead of 1, so the projection budget has to be raised.

```python
opt = TRexOptions()
opt.tFDR = 0.1
opt.solver = SolverType.AFS_Multinomial
opt.max_vd_proj = 600            # default budget runs dry with C-1 axes per step
result = TRexSelector(opt).run(X, labels.astype(float))
```

Centering the labels produces silent nonsense. Pass them raw.

## API reference

### Solvers

All solvers share one constructor and one set of accessors.

| Class | Response | Score `s` |
|-------|----------|-----------|
| `VD_LARS` | Gaussian | least-squares residual, equiangular step |
| `VD_OMP` | Gaussian | least-squares residual, full OLS refit |
| `VD_AFS` | Gaussian | least-squares residual, damped refit (`rho`) |
| `VD_AFS_Logistic` | Bernoulli | `y - sigmoid(eta)` |
| `VD_AFS_Poisson` | count | `y - exp(eta)` |
| `VD_AFS_Multinomial` | categorical, `C` classes | `y^(c) - softmax^(c)(eta)`, `c = 1..C-1` |
| `VD_AFS_Cox` | censored event time | martingale residual `delta - exp(eta) * Lambda_0(t)` |

```python
solver = VD_AFS_Cox(X, y, num_dummies, options)
```

- `X`: `(n, p)` Fortran-order float64, column-centered and unit-L2 normalized
- `y`: `(n,)` float64. Centered for Gaussian, raw for every other family
- `num_dummies`: `L`, typically a multiple of `p`
- `options`: a `VDOptions` instance

Shared methods: `run(T)`, `beta_real()`, `beta_view_copy()`,
`active_features()`, `active_indices()`, `vd_corr()`, `vd_proj()`,
`vd_stick()`, `is_dummy_realized()`, `normx()`, `basis_size()`,
`n_samples()`, `n_features()`, `num_dummies()`, `num_realized_dummies()`.

### VDOptions

```python
opt = VDOptions()
opt.T_stop = 100          # stop once this many dummies are realized
opt.max_vd_proj = 100     # basis vectors tracked for the VD projections
opt.rho = 1.0             # 1.0 = OMP, in (0,1) = AFS
opt.eps = 1e-12
opt.standardize = False   # True: center and normalize X internally
opt.seed = 0
opt.dummy_law = VDDummyLaw.Spherical   # or VDDummyLaw.Gaussian
opt.cox_status = ...      # (n,) float64 in {0,1}, AFS_Cox only
opt.cox_efron = False     # False = Breslow ties, True = Efron
opt.mmap_fd = -1          # file descriptor for pread I/O, -1 = off
opt.mmap_block_cols = 0   # columns per pread block, 0 = auto
opt.debug = False
```

`cox_status`, `cox_efron` and the four GLM `SolverType` values are what this
repository adds. Everything else is upstream.

### TRexSelector

```python
opt = TRexOptions()
opt.tFDR = 0.1
opt.K = 20                # random experiments
opt.L_factor = 5          # L = L_factor * p
opt.T_stop = -1           # -1 = auto, min(L, n/2)
opt.solver = SolverType.AFS_Cox
opt.calib = CalibMode.CalibrateT
opt.rho = 0.1
opt.max_vd_proj = 100
opt.n_threads = 0         # 0 = auto, OpenMP when available
opt.stride_width = 1
opt.posthoc_mode = False
opt.max_stale_strides = 3
opt.verbose = True
opt.seed = 42
```

`TRexResult` carries `selected_var`, `v_thresh`, `T_stop`, `num_dummies`,
`L_calibrated`, `V`, `FDP_hat_mat`, `Phi_mat`, `Phi_prime` and `K`.

Calibration modes: `FixedTL` (v only), `CalibrateT` (T and v at fixed L),
`CalibrateL` (scan L), `CalibrateBoth`.

## Preprocessing requirements

The solvers assume `X` is column-centered, unit-L2 normalized and in Fortran
order. The response is centered for the Gaussian families and raw for all
others.

```python
def preprocess_X(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    X -= X.mean(axis=0, keepdims=True)
    X /= np.linalg.norm(X, axis=0, keepdims=True).clip(min=eps)
    return np.asfortranarray(X)
```

## Repository structure

```
virtual-dummies-glm/
├── CMakeLists.txt            # vd_selectors + the opt-in benchmark targets
├── pyproject.toml            # scikit-build-core config
├── NOTICE                    # file-by-file provenance
├── src/                      # C++ core
│   ├── vd_common.hpp         # shared types and options
│   ├── vd_base.hpp/cpp       # dummy pool, basis growth, Cholesky, accessors
│   ├── vd_lars.hpp/cpp       # upstream Gaussian solvers
│   ├── vd_omp.hpp/cpp
│   ├── vd_afs.hpp/cpp
│   ├── vd_afs_logistic.hpp/cpp     # Bernoulli
│   ├── vd_afs_poisson.hpp/cpp      # count
│   ├── vd_afs_multinomial.hpp/cpp  # categorical
│   ├── vd_afs_cox.hpp/cpp          # censored event time
│   ├── trex.hpp/cpp          # calibration and solver dispatch
│   ├── memory_mapped_eigen_matrix.hpp
│   └── bindings.cpp          # pybind11
├── vd_selectors/             # Python package. The compiled module lands here
│   └── __init__.py           # at install time, see "Where the code lives"
├── extern/eigen-5.0.0/       # vendored Eigen
├── experiments/              # one folder per subsection, see experiments/README.md
│   ├── sec4_1_equivalence/   # SNR sweeps, AD baselines, knockoff arm, Fig. 1
│   ├── sec4_2_memory_runtime/ # memory and runtime benchmark
│   ├── sec4_3_hapnest/       # application 1, disease subtype classification
│   └── sec4_4_kirc/          # application 2, renal carcinoma survival
└── results/                  # the numbers behind the paper, see results/README.md
    ├── sec4_1_equivalence/   # summary per model + Fig. 1
    ├── sec4_2_memory_runtime/
    ├── sec4_3_hapnest/       # Table 2
    └── sec4_4_kirc/          # Table 3
```

## Where the code lives

`vd_selectors/` holds only `__init__.py`, which re-exports names. That's by
design and it's how the upstream project is laid out too: the solvers are C++
in `src/`, compiled into one pybind11 module. `pip install .` builds it and
CMake drops `vd_selectors.<abi>.so` next to that `__init__.py`, so the import
works from there:

```
CMakeLists.txt:  install(TARGETS vd_selectors LIBRARY DESTINATION vd_selectors)
```

Before the first build the folder therefore looks nearly empty. After
`pip install -e .` the module is in place and `import vd_selectors` resolves to
the compiled extension. To read the implementation of a solver, open
`src/vd_afs_cox.cpp` and friends.

## Reproducing the paper

[`experiments/README.md`](experiments/README.md) has the script-to-result map,
the exact settings and what runs where. Short version:

```bash
pip install -e . && pip install -r experiments/requirements.txt
python experiments/sec4_1_equivalence/sim_snr_poisson.py    # and the other three families
python experiments/sec4_1_equivalence/plot_snr_panel.py
```

The SNR sweeps take hours on 96 cores. The results they produce are already in
`results/`, so `plot_snr_panel.py` redraws the figure in a second.

The HAPNEST genotypes for Sec. 4.3 are on Harvard Dataverse
([doi:10.7910/DVN/RZ3FZT](https://doi.org/10.7910/DVN/RZ3FZT)), deposited by
Koka et al. `experiments/sec4_3_hapnest/prepare_hapnest_small.sh` downloads and
preprocesses them.
The TCGA-KIRC data for Sec. 4.4 is downloaded and preprocessed by
`experiments/sec4_4_kirc/kirc_data.py`.

## Attribution

See [NOTICE](NOTICE). In one paragraph: the virtual dummy construction, the
Gaussian solvers, the T-Rex implementation, the memory-mapped matrix, the build
system and the HAPNEST genotype pipeline are the work of Taulant Koka, Jasin
Machkour, Daniel P. Palomar and Michael Muma. The score-based extension, the
four GLM and Cox solvers, the explicitly augmented GLM baselines, the
simulation study, the memory benchmarks and the two applications were written
for this paper.

## References

- **Virtual dummies**: Koka, T., Machkour, J., Palomar, D. P., & Muma, M.
  (2026). Virtual Dummies: Enabling Scalable FDR-Controlled Variable Selection
  via Sequential Sampling of Null Features.
- **T-Rex selector**: Machkour, J., Muma, M., & Palomar, D. P. (2025). The
  terminating-random experiments selector: Fast high-dimensional variable
  selection with false discovery rate control. *Signal Processing*, 231, 109894.
- **OMP**: Pati, Y. C., Rezaiifar, R., & Krishnaprasad, P. S. (1993).
  Orthogonal matching pursuit. *Proc. 27th Asilomar Conf.*
- **LARS**: Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004).
  Least angle regression. *The Annals of Statistics*, 32(2).
- **Adaptive forward stepwise**: Zhang, I., & Tibshirani, R. (2026). Adaptive
  Forward Stepwise: A Method for High Sparsity Regression. *JMLR*, 27(35).
- **Model-X knockoffs**: Candès, E., Fan, Y., Janson, L., & Lv, J. (2018).
  Panning for gold: model-X knockoffs for high-dimensional controlled variable
  selection. *JRSS-B*, 80(3).
- **CoxKnockoff**: Li, Y., Yu, M., & Zhao, S. (2023). CoxKnockoff. *Stat*.
- **HAPNEST**: Wharrie, S., et al. (2023). HAPNEST: efficient, large-scale
  generation and evaluation of synthetic datasets for genotypes and phenotypes.
  *Bioinformatics*, 39(9).

## Citation

```bibtex
@inproceedings{mehler2026fdrglm,
  title={{FDR}-Controlled Variable Selection for Generalized Linear Models with
         Virtual Dummies},
  author={Mehler, Helena and Koka, Taulant and Muma, Michael},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal
             Processing (ICASSP)},
  year={2026},
}
```

Please also cite the work this builds on:

```bibtex
@article{koka2026virtualdummies,
  title={Virtual Dummies: Enabling Scalable {FDR}-Controlled Variable Selection
         via Sequential Sampling of Null Features},
  author={Koka, Taulant and Machkour, Jasin and Palomar, Daniel and Muma, Michael},
  journal={arXiv preprint},
  year={2026},
}

@article{machkour2025trex,
  title={The terminating-random experiments selector: Fast high-dimensional
         variable selection with false discovery rate control},
  author={Machkour, Jasin and Muma, Michael and Palomar, Daniel P.},
  journal={Signal Processing},
  volume={231},
  pages={109894},
  year={2025},
}
```

## Funding

H. Mehler was funded by the German Research Foundation (DFG), project number
550090872. T. Koka and M. Muma were funded by the ERC Starting Grant
ScReeningData, grant number 101042407.

Robust Data Science Group, TU Darmstadt.
