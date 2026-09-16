#!/usr/bin/env python
"""
Condense the per-replicate JSONs of the HAPNEST benchmark into Table 2.

    python experiments/sec4_3_hapnest/make_table2.py \
        results/sec4_3_hapnest/rerun/results --out results/sec4_3_hapnest/rerun

Takes one or more directories of run_<id>_<method>.json, as written by
benchmark_worker.py; pass the knockoff directory as a second one if the
knockoff rows ran elsewhere. Replicates that ended in an error are dropped and
reported. Writes table2.csv: one row per Table 2 method with the number of
replicates, mean FDP, TPP and number of selected SNPs with their standard
errors, and the mean runtime per replicate.
"""
# NEW in this repository (Mehler, Koka, Muma, 2026). GPLv3, see NOTICE.
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Table 2 rows, in the order of the paper
ROWS = [
    ("trex_multinom", "VD-Multinomial", "FDR"),
    ("trex_logit_ovr", "VD-Bernoulli, OvR", "FDR"),
    ("trex_gauss", "VD-Gaussian on labels", "FDR"),
    ("ko_mx_multinom", "Model-X Knockoffs", "FDR"),
    ("bh_multinom", "Multinomial Score + BH", "FDR, marginal"),
    ("by_multinom", "Multinomial Score + BY", "FDR, marginal"),
]


def se(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = {}
    for d in args.result_dirs:
        for f in sorted(Path(d).glob("run_*_*.json")):
            r = json.load(open(f))
            key = (r["run"], r["method"])
            # a later directory wins, unless its record is an error
            if key not in recs or "error" in recs[key] or "error" not in r:
                recs[key] = r
    df = pd.DataFrame(recs.values())
    if df.empty:
        raise SystemExit("no run_*_*.json in " + ", ".join(args.result_dirs))

    if "error" in df:
        bad = df[df["error"].notna()]
        for _, r in bad.iterrows():
            print(f"  dropped run {r['run']} {r['method']}: {r['error']}")
        df = df[df["error"].isna()]

    rows = []
    for key, label, crit in ROWS:
        g = df[df["method"] == key]
        if g.empty:
            print(f"  missing: {key}")
            continue
        rows.append({
            "method": label, "key": key, "error_criterion": crit,
            "n_reps": len(g),
            "fdp": g["fdp"].mean(), "fdp_se": se(g["fdp"]),
            "tpp": g["tpp"].mean(), "tpp_se": se(g["tpp"]),
            "n_selected": g["n_disc"].mean(), "n_selected_se": se(g["n_disc"]),
            "runtime_min": g["runtime_s"].mean() / 60,
        })
    table = pd.DataFrame(rows).round(3)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table2.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
