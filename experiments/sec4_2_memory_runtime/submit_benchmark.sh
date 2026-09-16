#!/bin/bash
# NEW in this repository (Mehler, Koka, Muma, 2026). GPLv3, see NOTICE.
# One memory/runtime benchmark of Sec. 4.2. Build first, then submit from the
# repository root:
#
#   cmake -B build -DBUILD_BENCHMARKS=ON
#   cmake --build build --target bench_memory_poisson -j
#   mkdir -p logs
#   sbatch experiments/sec4_2_memory_runtime/submit_benchmark.sh poisson
#
# Replace <project>. 20 cores are requested for the 76 GB of memory, the
# benchmark itself runs single-threaded.
#
#SBATCH -J bench_mem
#SBATCH -A <project>
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --mem-per-cpu=3800
#SBATCH -t 23:00:00
#SBATCH -o logs/bench_%x_%j.out
#SBATCH -e logs/bench_%x_%j.err

set -euo pipefail
FAMILY="${1:?usage: sbatch submit_benchmark.sh logistic|poisson|multinomial|cox}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MKL_THREADING_LAYER=SEQUENTIAL

export VD_MEM_CAP_GB="${VD_MEM_CAP_GB:-70}"
case "$FAMILY" in
  logistic)    export ADLOGIT_MEM_CAP_GB="${ADLOGIT_MEM_CAP_GB:-70}" ;;
  poisson)     export ADPOIS_MEM_CAP_GB="${ADPOIS_MEM_CAP_GB:-70}" ;;
  multinomial) export ADMULTI_MEM_CAP_GB="${ADMULTI_MEM_CAP_GB:-70}" ;;
  cox)         export ADCOX_MEM_CAP_GB="${ADCOX_MEM_CAP_GB:-70}" ;;
  *) echo "unknown family $FAMILY"; exit 1 ;;
esac

echo "node: $(hostname)  start: $(date)"
"./build/bench_memory_${FAMILY}"
python experiments/sec4_2_memory_runtime/summarize_benchmark.py "results/sec4_2_memory_runtime/${FAMILY}"
echo "end: $(date)"
