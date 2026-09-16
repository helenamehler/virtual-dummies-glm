#!/usr/bin/env python
"""
Runs the Table 2 methods on ONE HAPNEST replicate.

    python benchmark_worker.py <data_dir> <pheno_dir> <run_id> \
        <methods_file> <res_dir> [cfg_json]

Writes one JSON per method to <res_dir>/run_<run_id>_<method>.json with FDP,
TPP, the selected SNPs, runtime and peak memory. make_table2.py condenses them.

Methods
-------
    trex_multinom     VD-Multinomial, the proposed selector
    trex_logit_ovr    VD-Bernoulli one-vs-rest at alpha/(C-1), unioned
    trex_gauss        VD-Gaussian (OMP) on the centered integer labels
    bh_multinom       multinomial score test + BH
    by_multinom       multinomial score test + BY
    ko_mx_multinom    model-X knockoffs, see knockoff_multinomial.py
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), adapted from the binary HAPNEST pipeline of Koka et al. (2026). GPLv3, see NOTICE.

from __future__ import annotations

import json
import os
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multinomial_baselines import bh_by_select, multinomial_score_pvalues
from knockoff_multinomial import KO_DISPATCH

warnings.filterwarnings("ignore")

ALL_METHODS = [
    "trex_multinom", "trex_logit_ovr", "trex_gauss",
    "bh_multinom", "by_multinom", "ko_mx_multinom",
]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def peak_rss_mb():
    try:
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    except Exception:
        return -1.0


def fdp_tpp(selected, truth):
    sel = set(map(int, np.asarray(selected, dtype=int).ravel()))
    tru = set(map(int, np.asarray(truth, dtype=int).ravel()))
    tp, fp, R, P = len(sel & tru), len(sel - tru), len(sel), len(tru)
    return (fp / R if R else 0.0), (tp / P if P else 0.0)


# =============================================================
# Data loading: pread -> memmap -> npz, same cascade as the binary worker
# =============================================================

def load_genotypes(data_dir, run_id):
    run_dir = Path(data_dir) / f"run_{run_id}"
    meta_p, xstd, xraw = (run_dir / "meta.json", run_dir / "X_std.dat",
                          run_dir / "X_raw.dat")
    npz_p = run_dir / "data.npz"
    holders, mmap_fd, mode = [], -1, "npz"

    if xstd.exists() and xraw.exists() and meta_p.exists():
        try:
            from vd_selectors import MMapMatrix
            meta = json.load(open(meta_p))
            n, p = meta["n_samples"], meta["p_pruned_total"]
            mm_std = MMapMatrix(str(xstd), n, p, writable=False)
            mm_raw = MMapMatrix(str(xraw), n, p, writable=False)
            holders += [mm_std, mm_raw]
            Xc, X_raw = mm_std.as_array(), mm_raw.as_array()
            mmap_fd, mode = mm_std.fileno(), "pread"
        except Exception as e:
            log(f"MMapMatrix unavailable ({e}), falling back to memmap")
            holders.clear()
            mmap_fd = -1

    if mmap_fd < 0 and xraw.exists() and meta_p.exists():
        meta = json.load(open(meta_p))
        n, p = meta["n_samples"], meta["p_pruned_total"]
        X_raw = np.memmap(xraw, dtype="float64", mode="r", shape=(n, p), order="F")
        Xc = np.array(X_raw, dtype=np.float64, order="F")
        Xc -= Xc.mean(axis=0, keepdims=True)
        nrm = np.linalg.norm(Xc, axis=0, keepdims=True)
        nrm[nrm == 0] = 1.0
        Xc /= nrm
        mode = "memmap"
    elif mode == "npz":
        if not npz_p.exists():
            return None
        d = np.load(npz_p)
        Xc = np.asarray(d["X"], dtype=np.float64)
        X_raw = np.asarray(d["X_raw"], dtype=np.float64)
        d.close()

    return Xc, X_raw, mmap_fd, mode, holders


def load_phenotype(pheno_dir, run_id):
    path = Path(pheno_dir) / f"pheno_run_{run_id}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    out = dict(
        y=np.asarray(d["y"], dtype=np.float64),
        causal_idx=np.asarray(d["causal_idx"], dtype=int),
        is_shared=np.asarray(d["is_shared"], dtype=bool),
        causal_class=np.asarray(d["causal_class"], dtype=int),
        C=int(d["C"]),
    )
    d.close()
    return out


# =============================================================
# T-Rex variants
# =============================================================

def _gauss_solver(cfg):
    """Solver behind the Gaussian row: OMP."""
    return cfg.get("gauss_solver", "OMP")


def _base_opts(cfg, mmap_fd, solver_name, gaussian):
    from vd_selectors import (
        TRexOptions, SolverType, CalibMode, VDDummyLaw,
    )
    if not hasattr(SolverType, solver_name):
        raise RuntimeError(
            f"this vd_selectors build has no {solver_name}; rebuild with "
            f"`pip install -e .` (available: "
            f"{[s for s in dir(SolverType) if not s.startswith('_')]})")
    opt = TRexOptions()
    opt.tFDR = float(cfg["alpha"])
    opt.K = cfg.get("K", 20)
    opt.L_factor = cfg.get("L_factor", 5)
    opt.T_stop = cfg.get("T_stop", -1)
    opt.calib = getattr(CalibMode, cfg.get("calib", "CalibrateT"))
    opt.solver = getattr(SolverType, solver_name)
    opt.dummy_law = VDDummyLaw.Spherical
    opt.posthoc_mode = cfg.get("posthoc_mode", False)
    opt.n_threads = cfg.get("n_threads", 1)
    opt.verbose = False
    opt.seed = cfg.get("seed", 42)
    if gaussian:
        opt.stride_width = cfg.get("stride_width_gauss", 5)
        opt.max_stale_strides = cfg.get("max_stale_strides_gauss", 2)
    else:
        opt.rho = cfg.get("rho", 0.1)
        opt.stride_width = cfg.get("stride_width", 1)
        opt.max_stale_strides = cfg.get("max_stale_strides", 999)
    if mmap_fd >= 0:
        opt.mmap_fd = mmap_fd
        opt.mmap_block_cols = cfg.get("mmap_block_cols", 512)
    return opt


def _run_trex(Xc, y_vec, cfg, mmap_fd, solver_name, gaussian, extra_opts=None):
    from vd_selectors import TRexSelector
    opt = _base_opts(cfg, mmap_fd, solver_name, gaussian)
    for k, v in (extra_opts or {}).items():
        setattr(opt, k, v)
    Xf = Xc if mmap_fd >= 0 else np.asfortranarray(Xc, dtype=np.float64)
    res = TRexSelector(opt).run(Xf, np.ascontiguousarray(y_vec, dtype=float))
    return np.asarray(res.selected_var, dtype=int), {
        "T_stop": int(getattr(res, "T_stop", 0)),
        "L": int(getattr(res, "num_dummies", 0)),
        "L_calibrated": int(getattr(res, "L_calibrated", 0)),
    }


def m_trex_multinom(Xc, X_raw, y, cfg, mmap_fd):
    """The proposed selector. Raw integer labels, enlarged projection budget."""
    y_lab = np.asarray(y, dtype=float).ravel()
    if y_lab.min() < 0 or not np.allclose(y_lab, np.round(y_lab)):
        raise ValueError("multinomial solver needs raw integer labels 0..C-1")
    sel, ex = _run_trex(
        Xc, y_lab, cfg, mmap_fd, "AFS_Multinomial", gaussian=False,
        # The revealed subspace grows by up to C-1 axes per step, not one.
        extra_opts={"max_vd_proj": cfg.get("max_vd_proj", 600)})
    ex["max_vd_proj"] = cfg.get("max_vd_proj", 600)
    return sel, ex


def m_trex_gauss(Xc, X_raw, y, cfg, mmap_fd):
    """Gaussian OMP on the centered integer labels: invents a class ordering."""
    yv = np.asarray(y, dtype=float).ravel()
    solver = _gauss_solver(cfg)
    sel, ex = _run_trex(Xc, yv - yv.mean(), cfg, mmap_fd, solver, gaussian=True)
    ex["gauss_solver"] = solver
    return sel, ex


def _ovr(Xc, y, cfg, mmap_fd, solver_name, gaussian, extra_opts=None):
    y = np.asarray(y, dtype=int).ravel()
    C = int(y.max()) + 1
    K = C - 1
    sel_all, per_class = set(), {}
    for c in range(1, C):
        cfg_c = dict(cfg, alpha=float(cfg["alpha"]) / K)
        yc = (y == c).astype(float)
        if gaussian:
            yc = yc - yc.mean()
        sel_c, _ = _run_trex(Xc, yc, cfg_c, mmap_fd, solver_name, gaussian,
                             extra_opts)
        per_class[f"class_{c}"] = len(sel_c)
        sel_all |= set(sel_c.tolist())
    return np.array(sorted(sel_all), dtype=int), {
        "alpha_per_class": float(cfg["alpha"]) / K,
        "per_class_n": json.dumps(per_class), "n_runs": K}


def m_trex_logit_ovr(Xc, X_raw, y, cfg, mmap_fd):
    """The honest strong competitor: correctly specified per class."""
    return _ovr(Xc, y, cfg, mmap_fd, "AFS_Logistic", gaussian=False,
                extra_opts={"max_vd_proj": cfg.get("max_vd_proj_logistic", 300)})


# =============================================================
# Marginal methods
# =============================================================

def m_bh_multinom(Xc, X_raw, y, cfg, mmap_fd):
    pv, info = multinomial_score_pvalues(X_raw, y, C=cfg.get("C"))
    return bh_by_select(pv, cfg["alpha"], "fdr_bh"), info


def m_by_multinom(Xc, X_raw, y, cfg, mmap_fd):
    pv, info = multinomial_score_pvalues(X_raw, y, C=cfg.get("C"))
    return bh_by_select(pv, cfg["alpha"], "fdr_by"), info


DISPATCH = {
    "trex_multinom": m_trex_multinom,
    "trex_logit_ovr": m_trex_logit_ovr,
    "trex_gauss": m_trex_gauss,
    "bh_multinom": m_bh_multinom,
    "by_multinom": m_by_multinom,
    **KO_DISPATCH,
}


# =============================================================
# Driver
# =============================================================

def run_one(Xc, X_raw, pheno, methods, cfg, res_dir, run_id, mmap_fd, load_mode):
    n, p = Xc.shape
    y = pheno["y"]
    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg)
    cfg.setdefault("C", pheno["C"])
    cfg.setdefault("_run_id", run_id)      # keys the knockoff cache

    counts = np.bincount(y.astype(int), minlength=pheno["C"]).tolist()
    log(f"run={run_id} n={n} p={p} C={pheno['C']} s={len(pheno['causal_idx'])} "
        f"({int(pheno['is_shared'].sum())} shared) "
        f"counts={counts} load={load_mode}")

    summary = []
    for method in methods:
        rec = {
            "run": run_id, "method": method, "n": n, "p": p, "C": pheno["C"],
            "alpha": cfg["alpha"], "n_causal": len(pheno["causal_idx"]),
            "n_shared": int(pheno["is_shared"].sum()),
            "class_counts": json.dumps(counts), "load_mode": load_mode,
        }
        t0 = time.perf_counter()
        try:
            sel, extra = DISPATCH[method](Xc, X_raw, y, cfg, mmap_fd)
            fdp, tpp = fdp_tpp(sel, pheno["causal_idx"])
            rec.update(extra)
            rec.update(fdp=fdp, tpp=tpp, n_disc=int(len(sel)),
                       selected=sorted(int(j) for j in sel))
            log(f"  {method:16s} R={len(sel):5d} FDP={fdp:.3f} TPP={tpp:.3f}")
            summary.append((method, len(sel), fdp, tpp))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            log(f"  {method:16s} ERROR {rec['error']}")
            log(traceback.format_exc(limit=3))
        rec["runtime_s"] = round(time.perf_counter() - t0, 3)
        rec["peak_rss_mb"] = round(peak_rss_mb(), 1)
        with open(res_dir / f"run_{run_id}_{method}.json", "w") as f:
            json.dump(rec, f, indent=1)

    return summary


def main():
    data_dir, pheno_dir = sys.argv[1], sys.argv[2]
    run_id = int(sys.argv[3])
    methods_file, res_dir = sys.argv[4], sys.argv[5]
    cfg = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
    cfg.setdefault("alpha", 0.1)

    methods = [m.strip() for m in open(methods_file) if m.strip()]
    unknown = [m for m in methods if m not in DISPATCH]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}\nknown: {ALL_METHODS}")

    pheno = load_phenotype(pheno_dir, run_id)
    if pheno is None:
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        for m in methods:
            json.dump({"run": run_id, "method": m, "error": "pheno not found"},
                      open(Path(res_dir) / f"run_{run_id}_{m}.json", "w"))
        raise SystemExit(f"phenotype for run {run_id} not found in {pheno_dir}")

    loaded = load_genotypes(data_dir, run_id)
    if loaded is None:
        raise SystemExit(f"genotypes for run {run_id} not found in {data_dir}")
    Xc, X_raw, mmap_fd, load_mode, holders = loaded

    needs_ram = any(m.startswith(("bh", "by", "ko")) for m in methods)
    if needs_ram and load_mode in ("pread", "memmap"):
        log("loading X_raw into RAM for the marginal tests")
        X_raw = np.array(X_raw, dtype=np.float64, order="F")

    run_one(Xc, X_raw, pheno, methods, cfg, res_dir, run_id, mmap_fd, load_mode)
    del holders


if __name__ == "__main__":
    main()
