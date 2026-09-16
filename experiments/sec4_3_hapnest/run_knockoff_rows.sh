#!/usr/bin/env bash
# NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
# Driver for the model-X knockoff row of Table 2 (Sec. 4.3).
#
#   bash run_knockoff_rows.sh pilot   # 3 replicates
#   bash run_knockoff_rows.sh full    # all N_REPS replicates
#
# Same OUT_ROOT and phenotypes as run_multinomial_benchmark.sh, so the JSONs
# land next to the other methods. Afterwards
#   bash run_multinomial_benchmark.sh table

set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-../../data/hapnest_preprocessed_small}"
OUT_ROOT="${OUT_ROOT:-../../results/sec4_3_hapnest/rerun}"
N_REPS="${N_REPS:-50}"
ALPHA="${ALPHA:-0.1}"
PY="${PY:-python3}"

BENCH_DIR="3_4_benchmark"
PHENO_DIR="${OUT_ROOT}/phenotypes"
RES_DIR="${OUT_ROOT}/results"
KO_CACHE="${KO_CACHE:-${OUT_ROOT}/knockoffs}"
METHODS_FILE="${OUT_ROOT}/methods_knockoff.txt"

# Must match run_multinomial_benchmark.sh exactly, or the two halves of the
# table are computed on different phenotypes.
PHENO_CFG="${PHENO_CFG:-$(cat <<'JSON'
{"C":4, "s":12, "n_shared":4,
 "class_prev":[0.5,0.16667,0.16667,0.16666],
 "het_rr_shared":[1.15,1.30], "het_rr_specific":[1.20,1.42],
 "seed0":42}
JSON
)}"

# The construction is knockpy's: ko_method=null means knockpy's default, MVR,
# with a Ledoit-Wolf covariance estimate.
KO_CFG="${KO_CFG:-$(cat <<JSON
{"alpha":${ALPHA}, "C":4,
 "ko_cache_dir":"${KO_CACHE}",
 "ko_method":null, "ko_shrinkage":"ledoitwolf",
 "ko_how_approx":"blockdiag", "ko_max_block":1000,
 "ko_num_processes":${KO_N_JOBS:-1}, "ko_seed":123,
 "ko_fstat":"multinomial_l1", "ko_agg":"l2",
 "ko_n_lambda":3, "ko_cv":3, "ko_n_jobs":${KO_N_JOBS:-1},
 "ko_max_iter":200, "ko_tol":0.001}
JSON
)}"

mkdir -p "$PHENO_DIR" "$RES_DIR" "$KO_CACHE"
cat > "$METHODS_FILE" <<'EOF'
ko_mx_multinom
EOF

gen_pheno () {
  echo "== phenotypes for $1 replicates"
  for i in $(seq 1 "$1"); do
    $PY "$BENCH_DIR/generate_phenotype.py" \
        "$DATA_DIR" "$PHENO_DIR" "$i" "$PHENO_CFG"
  done
}

gen_knockoffs () {
  echo "== knockoff matrices for $1 replicates into $KO_CACHE"
  for i in $(seq 1 "$1"); do
    $PY "$BENCH_DIR/knockoff_multinomial.py" generate \
        "$DATA_DIR" "$i" "$KO_CACHE" "$KO_CFG"
  done
}

bench () {
  echo "== knockoff rows on $1 replicates"
  for i in $(seq 1 "$1"); do
    echo "-- replicate $i"
    $PY "$BENCH_DIR/benchmark_worker.py" \
        "$DATA_DIR" "$PHENO_DIR" "$i" "$METHODS_FILE" "$RES_DIR" "$KO_CFG"
  done
}

case "${1:-pilot}" in
  pilot)
    gen_pheno 3
    gen_knockoffs 3
    bench 3
    ;;
  full)
    gen_pheno "$N_REPS"
    gen_knockoffs "$N_REPS"
    bench "$N_REPS"
    ;;
  *)
    echo "usage: $0 {pilot|full}" >&2
    exit 1
    ;;
esac
