#!/usr/bin/env bash
# NEW in this repository (Mehler, Koka, Muma, 2026). GPLv3, see NOTICE.
# Genotypes for Application 1 (Sec. 4.3): download, arrange, preprocess, check.
#
#   bash prepare_hapnest_small.sh download     # ~24 GB from Harvard Dataverse
#   bash prepare_hapnest_small.sh arrange      # run_<id>/ folders via symlinks
#   bash prepare_hapnest_small.sh preprocess   # LD pruning + standardization
#   bash prepare_hapnest_small.sh check        # n, p and max |corr| per run
#   bash prepare_hapnest_small.sh all          # the four in a row
#
# Environment:
#   RAW_DIR    downloaded PLINK files     default ../../data/hapnest_raw_small
#   DATA_DIR   preprocessed output        default ../../data/hapnest_preprocessed_small
#   RUNS       replicates, "1-100" or "1-3,7"                    default 1-100
#   PY         python interpreter                                default python3
#
# Inside a Slurm array job (SLURM_ARRAY_TASK_ID set), `preprocess` handles
# only that one replicate. See submit_preprocess_small.sh.
#
# Disk: every preprocessed replicate is two dense float64 matrices of
# 10 000 x ~31 000, about 5 GB. All 100 need about 500 GB. Memory: about 24 GB
# per replicate while preprocessing.

set -euo pipefail
cd "$(dirname "$0")"

RAW_DIR="${RAW_DIR:-../../data/hapnest_raw_small}"
DATA_DIR="${DATA_DIR:-../../data/hapnest_preprocessed_small}"
RUNS="${RUNS:-1-100}"
PY="${PY:-python3}"
PRE="2_preprocessing"

expand_runs() {  # "1-3,7" -> 1 2 3 7
  local part a b out=()
  IFS=',' read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    a="${part%-*}"; b="${part#*-}"
    out+=($(seq "$a" "$b"))
  done
  echo "${out[@]}"
}

last_run() { expand_runs "$RUNS" | awk '{print $NF}'; }

do_download() {
  $PY "$PRE/download_hapnest_small.py" "$RAW_DIR" --only-runs "$RUNS"
}

do_arrange() {
  $PY "$PRE/arrange_hapnest_runs.py" "$RAW_DIR"
}

do_preprocess() {
  mkdir -p "$DATA_DIR"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  local ids
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    ids="$SLURM_ARRAY_TASK_ID"
  else
    ids="$(expand_runs "$RUNS")"
  fi
  for rid in $ids; do
    if [[ -f "$DATA_DIR/run_${rid}/meta.json" ]]; then
      echo "run $rid: already preprocessed, skipping"
      continue
    fi
    $PY -u "$PRE/preprocess_hapnest_small.py" "$RAW_DIR" "$DATA_DIR" \
        "$(last_run)" --run-id "$rid"
  done
}

do_check() {
  $PY - "$DATA_DIR" $(expand_runs "$RUNS") <<'PYEOF'
import json, sys
from pathlib import Path
root, ids = Path(sys.argv[1]), [int(i) for i in sys.argv[2:]]
missing, ps = [], []
for i in ids:
    m = root / f"run_{i}" / "meta.json"
    if not m.exists():
        missing.append(i)
        continue
    d = json.load(open(m))
    ps.append(d["p_pruned_total"])
    ok = d["n_samples"] == 10000 and d["max_corr_after_pruning"] <= 0.7
    print(f"run {i:3d}: n = {d['n_samples']}, p = {d['p_pruned_total']}, "
          f"max |corr| = {d['max_corr_after_pruning']:.3f}"
          + ("" if ok else "   <-- check"))
if ps:
    print(f"\n{len(ps)} replicates, p from {min(ps)} to {max(ps)}")
if missing:
    print(f"missing: {missing}")
    sys.exit(1)
PYEOF
}

case "${1:-}" in
  download)   do_download ;;
  arrange)    do_arrange ;;
  preprocess) do_preprocess ;;
  check)      do_check ;;
  all)        do_download; do_arrange; do_preprocess; do_check ;;
  *) sed -n '3,22p' "$0"; exit 1 ;;
esac
