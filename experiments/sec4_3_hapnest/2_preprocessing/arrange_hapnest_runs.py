#!/usr/bin/env python
"""
Put downloaded HAPNEST files into the layout the preprocessing expects.

Dataverse delivers the replicates flat,

    RAW_DIR/run_17_chr-1.bed, run_17_chr-1.bim, run_17_chr-1.fam, ...

while preprocess_hapnest_small.py reads RAW_DIR/run_17/run_17_chr-1.bed, the
layout HAPNEST writes when the genotypes are simulated on the cluster. This
script creates the run_<id>/ folders and fills them with symlinks (no copies,
no extra disk). Files that already sit in run_<id>/ are left alone, so running
it twice, or on a simulated tree, does nothing.

Usage:
    python arrange_hapnest_runs.py RAW_DIR
"""
# NEW in this repository (Mehler, Koka, Muma, 2026). GPLv3, see NOTICE.
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"^run_(\d+)_chr-\d+\.(bed|bim|fam|sample)$")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    raw = Path(sys.argv[1]).resolve()
    if not raw.is_dir():
        sys.exit(f"not a directory: {raw}")

    linked, runs = 0, set()
    for f in sorted(raw.iterdir()):
        m = PATTERN.match(f.name)
        if not m or not f.is_file():
            continue
        run_dir = raw / f"run_{m.group(1)}"
        run_dir.mkdir(exist_ok=True)
        dst = run_dir / f.name
        if not dst.exists():
            dst.symlink_to(Path("..") / f.name)
            linked += 1
        runs.add(int(m.group(1)))

    nested = sorted(int(d.name[4:]) for d in raw.glob("run_*")
                    if d.is_dir() and d.name[4:].isdigit()
                    and (d / f"{d.name}_chr-1.bed").exists())
    print(f"{linked} symlinks created, {len(nested)} replicates ready "
          f"({nested[0] if nested else '-'} .. {nested[-1] if nested else '-'})")
    if not nested:
        sys.exit("no run_<id>_chr-1.bed found, check RAW_DIR")


if __name__ == "__main__":
    main()
