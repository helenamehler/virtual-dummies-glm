#!/usr/bin/env python3
"""The numbers of Sec. 4.2 from one benchmark raw.csv.

    python experiments/sec4_2_memory_runtime/summarize_benchmark.py results/sec4_2_memory_runtime/poisson

Medians over the replicates. Peak RSS in raw.csv is MiB; the overhead is peak
RSS minus the n*p*8 bytes of (X, y), printed in SI units.
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
import csv
import statistics as st
import sys

MIB = 1024.0 ** 2


def main(run_dir):
    with open(f"{run_dir}/raw.csv") as fh:
        rows = list(csv.DictReader(fh))
    good = [r for r in rows if int(r["ok_vd"]) and int(r["ok_ad"])]
    if not good:
        raise SystemExit("no complete replicate, check the memory caps")
    n, p, L = int(good[0]["n"]), int(good[0]["p"]), int(good[0]["L"])
    base = n * p * 8.0
    ovh_vd = st.median(float(r["mem_vd_mb"]) for r in good) * MIB - base
    ovh_ad = st.median(float(r["mem_ad_mb"]) for r in good) * MIB - base
    t_vd = st.median(float(r["time_vd"]) for r in good)
    t_ad = st.median(float(r["time_ad"]) for r in good)

    print(f"n={n}  p={p}  L={L // p}p  replicates={len(good)}/{len(rows)}")
    print(f"  (X, y)         {base / 1e9:8.2f} GB")
    print(f"  overhead VD    {ovh_vd / 1e6:8.1f} MB")
    print(f"  overhead AD    {ovh_ad / 1e9:8.2f} GB")
    print(f"  runtime  VD    {t_vd:8.1f} s")
    print(f"  runtime  AD    {t_ad:8.1f} s   (VD is {t_ad / t_vd:.2f}x faster)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1].rstrip("/"))
