#!/usr/bin/env python
"""
Condense the output of kirc_selection.py into the two files behind Table 3.

    python experiments/sec4_4_kirc/make_table3.py results/sec4_4_kirc/rerun \
        [results/sec4_4_kirc/rerun/knockoff] --out results/sec4_4_kirc/rerun

Reads sweep_<tag>.csv and diag_<tag>.csv from every run directory given (the
knockoff rows run separately, into their own directory) and writes

    table3.csv          one row per Table 3 method: label, error criterion,
                        number of selected genes, T-Rex calibration
    selected_genes.csv  the genes selected by the two VD rows, with their
                        marginal Cox z-score and correlations with log time and
                        the event indicator
"""
# NEW in this repository (Mehler, Koka, Muma, 2026). GPLv3, see NOTICE.
import argparse
from pathlib import Path

import pandas as pd

# Table 3 rows, in the order of the paper
ROWS = [
    ("bh_cox", "Marginal Cox + BH", "FDR, marginal"),
    ("by_cox", "Marginal Cox + BY", "FDR, marginal"),
    ("cox_lasso", "Cox Lasso, CV", "none"),
    ("vd_afs_cox", "VD-AFS-Cox", "FDR, joint"),
    ("vd_afs_ols", "VD-AFS-Gaussian on log t", "FDR, joint"),
    ("cox_knockoff", "Cox Knockoffs", "FDR, joint"),
]
VD_METHODS = ("vd_afs_cox", "vd_afs_ols")


def read_all(run_dirs, prefix):
    frames = []
    for d in run_dirs:
        for f in sorted(Path(d).glob(f"{prefix}_*.csv")):
            frames.append(pd.read_csv(f).assign(tag=f.stem[len(prefix) + 1:]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    sweep = read_all(args.run_dirs, "sweep")
    if sweep.empty:
        raise SystemExit("no sweep_*.csv in " + ", ".join(args.run_dirs))
    tags = sorted(sweep["tag"].unique())
    if len(tags) != 1:
        raise SystemExit(f"runs on different data: {tags}")
    print(f"data: {tags[0]}")

    rows = []
    for key, label, crit in ROWS:
        hit = sweep[(sweep["method"] == key)
                    & (sweep["alpha"].isna() | (sweep["alpha"] == args.alpha))]
        if hit.empty:
            print(f"  missing: {key}")
            continue
        r = hit.iloc[-1]
        vd = key in VD_METHODS
        rows.append({
            "method": label, "key": key, "error_criterion": crit,
            "n_selected": int(r["n_selected"]),
            "T_stop": int(r["T_stop"]) if vd else None,
            "L": int(r["L"]) if vd else None,
            "v_thresh": r["v_thresh"] if vd else None,
        })
    table = pd.DataFrame(rows).astype({"T_stop": "Int64", "L": "Int64"})
    table["v_thresh"] = table["v_thresh"].astype(float).round(3)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table3.csv", index=False)
    print(table.to_string(index=False))

    diag = read_all(args.run_dirs, "diag")
    if not diag.empty:
        genes = diag[diag["method"].isin(VD_METHODS) & (diag["alpha"] == args.alpha)]
        genes = genes[["method", "gene", "z_cox", "r_logt", "r_event"]].round(3)
        genes.to_csv(out / "selected_genes.csv", index=False)
        print(f"\n{len(genes)} selected genes -> {out / 'selected_genes.csv'}")


if __name__ == "__main__":
    main()
