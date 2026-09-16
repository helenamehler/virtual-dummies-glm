#!/usr/bin/env bash
# NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
# Driver for the T-Rex and marginal rows of Table 2 (Sec. 4.3).
#
#   bash run_multinomial_benchmark.sh pilot   # 3 replicates
#   bash run_multinomial_benchmark.sh full    # all N_REPS replicates
#   bash run_multinomial_benchmark.sh table   # table2.csv from the JSONs
#
# DATA_DIR is the preprocessed HAPNEST directory (run_<id>/X_std.dat etc.),
# see prepare_hapnest_small.sh.

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------- settings
DATA_DIR="${DATA_DIR:-../../data/hapnest_preprocessed_small}"
OUT_ROOT="${OUT_ROOT:-../../results/sec4_3_hapnest/rerun}"
N_REPS="${N_REPS:-100}"
ALPHA="${ALPHA:-0.1}"

# Phenotype: 4 classes, half controls, 12 causal loci of which 4 shared.
PHENO_CFG="${PHENO_CFG:-$(cat <<'JSON'
{"C":4, "s":12, "n_shared":4,
 "class_prev":[0.5,0.16667,0.16667,0.16666],
 "het_rr_shared":[1.15,1.30], "het_rr_specific":[1.20,1.42],
 "seed0":42}
JSON
)}"

# T-Rex settings. max_vd_proj is raised because the multinomial basis grows by
# up to C-1 axes per step.
BENCH_CFG="${BENCH_CFG:-$(cat <<JSON
{"alpha":${ALPHA}, "C":4, "K":20, "L_factor":5, "T_stop":40, "rho":1,
 "calib":"CalibrateT", "max_vd_proj":600, "max_vd_proj_logistic":300,
 "n_threads":1, "seed":42}
JSON
)}"

BENCH_DIR="3_4_benchmark"
PHENO_DIR="${OUT_ROOT}/phenotypes"
RES_DIR="${OUT_ROOT}/results"
METHODS_FILE="${OUT_ROOT}/methods.txt"
PY="${PY:-python3}"

mkdir -p "$PHENO_DIR" "$RES_DIR"
cat > "$METHODS_FILE" <<'EOF'
trex_multinom
trex_logit_ovr
trex_gauss
bh_multinom
by_multinom
EOF

# ---------------------------------------------------------------- helpers
check_solver () {
  echo "== checking the vd_selectors build"
  $PY - <<'EOF'
import sys
try:
    from vd_selectors import SolverType
except Exception as e:
    sys.exit(f"FAIL: cannot import vd_selectors ({e})\n"
             "      run `pip install -e .` in the repository root")
have = [s for s in dir(SolverType) if not s.startswith('_') and s[0].isupper()]
print("   solvers:", have)
missing = [s for s in ("AFS_Multinomial", "AFS_Logistic", "OMP") if s not in have]
if missing:
    sys.exit(f"FAIL: missing {missing}\n"
             "      the installed vd_selectors build predates the multinomial\n"
             "      solver -- rebuild with `pip install -e .`")
print("   ok")
EOF
}

gen_pheno () {
  local reps=$1
  echo "== generating $reps phenotypes into $PHENO_DIR"
  for i in $(seq 1 "$reps"); do
    $PY "$BENCH_DIR/generate_phenotype.py" \
        "$DATA_DIR" "$PHENO_DIR" "$i" "$PHENO_CFG"
  done
}

bench () {
  local reps=$1
  echo "== running the benchmark on $reps replicates"
  for i in $(seq 1 "$reps"); do
    echo "-- replicate $i"
    $PY "$BENCH_DIR/benchmark_worker.py" \
        "$DATA_DIR" "$PHENO_DIR" "$i" "$METHODS_FILE" "$RES_DIR" "$BENCH_CFG"
  done
}

table () {
  echo "== table2.csv"
  $PY make_table2.py "$RES_DIR" --out "$OUT_ROOT"
}

# ---------------------------------------------------------------- modes
case "${1:-pilot}" in
  pilot)
    check_solver
    gen_pheno 3
    bench 3
    table
    ;;
  full)
    check_solver
    gen_pheno "$N_REPS"
    bench "$N_REPS"
    table
    ;;
  table)
    table
    ;;
  *)
    echo "usage: $0 {pilot|full|table}" >&2
    exit 1
    ;;
esac
