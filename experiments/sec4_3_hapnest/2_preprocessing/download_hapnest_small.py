#!/usr/bin/env python
"""
Download the small-scale HAPNEST genotypes of Koka et al. (2026) from Harvard
Dataverse, doi:10.7910/DVN/RZ3FZT.

The deposit holds 100 replicates of chromosome 1, already quality controlled
with PLINK (MAF > 1%, call rate > 95%, HWE p > 1e-6): n = 10 000 individuals
and about 94 500 SNPs per replicate, stored as run_<id>_chr-1.{bed,bim,fam}.
About 24 GB in total.

The script fetches one file at a time, because the Dataverse bulk ZIP download
has a size limit. Each file is written to <name>.part and renamed only once it
is complete and its size matches what Dataverse reports, so an interrupted run
leaves no half files. Transient server errors are retried with backoff.
Restarting skips everything that is already there.

Usage:
    python download_hapnest_small.py OUT_DIR
    python download_hapnest_small.py OUT_DIR --only-runs 1-3   # a pilot
"""
# NEW in this repository (Mehler, Koka, Muma, 2026). The data is from Koka et al. (2026). GPLv3, see NOTICE.
import argparse
import os
import re
import sys
import time

import requests

DOI = "doi:10.7910/DVN/RZ3FZT"
BASE = "https://dataverse.harvard.edu"

MAX_RETRIES = 6          # per file
BACKOFF = 5              # seconds, doubles with every attempt
TRANSIENT = {500, 502, 503, 504}


def download_one(fid, dst, expected_size=None):
    tmp = dst + ".part"
    url = f"{BASE}/api/access/datafile/{fid}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as r:
                if r.status_code in TRANSIENT:
                    raise requests.HTTPError(f"{r.status_code} transient")
                r.raise_for_status()
                with open(tmp, "wb") as out:
                    for chunk in r.iter_content(1 << 20):
                        out.write(chunk)
            if expected_size and os.path.getsize(tmp) != expected_size:
                raise IOError(f"size {os.path.getsize(tmp)} != expected {expected_size}")
            os.replace(tmp, dst)
            return True
        except (requests.RequestException, IOError) as e:
            wait = BACKOFF * (2 ** (attempt - 1))
            print(f"    attempt {attempt}/{MAX_RETRIES} failed ({e}), waiting {wait}s",
                  flush=True)
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    return False


def parse_runs(spec):
    """'1-3,7' -> {1, 2, 3, 7}"""
    runs = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        runs.update(range(int(a), int(b or a) + 1))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="where the PLINK files go")
    ap.add_argument("--only-runs", default=None,
                    help="download only these replicates, e.g. 1-3 or 1,5,9. "
                         "Files that belong to no replicate are always fetched")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wanted = parse_runs(args.only_runs) if args.only_runs else None

    meta = requests.get(f"{BASE}/api/datasets/:persistentId/",
                        params={"persistentId": DOI}, timeout=60).json()
    files = meta["data"]["latestVersion"]["files"]

    failed = []
    for f in files:
        df = f["dataFile"]
        name = df["filename"]
        m = re.match(r"run_(\d+)_", name)
        if wanted is not None and m and int(m.group(1)) not in wanted:
            continue
        dst = os.path.join(args.out_dir, name)
        size = df.get("filesize")
        if os.path.exists(dst) and (not size or os.path.getsize(dst) == size):
            print(f"skip {name}")
            continue
        print(f"get  {name}  {(size or 0) / 1e6:.0f} MB", flush=True)
        if not download_one(df["id"], dst, size):
            print(f"    FAILED: {name}", flush=True)
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} file(s) failed. Run the script again, it only "
              f"fetches what is missing:")
        for n in failed:
            print(f"  - {n}")
        sys.exit(1)
    print("\nall files complete.")


if __name__ == "__main__":
    main()
