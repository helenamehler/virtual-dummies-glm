#!/bin/bash
# NEW in this repository (Mehler, Koka, Muma, 2026). GPLv3, see NOTICE.
# One SNR sweep of Sec. 4.1 on a 96-core node. Submit from the repository root:
#
#   mkdir -p logs
#   sbatch experiments/sec4_1_equivalence/submit_sweep.sh poisson
#   sbatch experiments/sec4_1_equivalence/submit_sweep.sh logistic   # cox, multinomial
#
# Replace <project>. The memory request is the whole node on purpose: the AD
# arm holds K augmented designs per worker.
#
#SBATCH -J snr_sweep
#SBATCH -A <project>
#SBATCH -n 1
#SBATCH -c 96
#SBATCH --mem-per-cpu=3800
#SBATCH -t 08:00:00
#SBATCH -o logs/snr_%x_%j.out
#SBATCH -e logs/snr_%x_%j.err

set -euo pipefail
FAMILY="${1:?usage: sbatch submit_sweep.sh logistic|poisson|cox|multinomial}"

# activate the environment with vd_selectors installed, e.g.
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate vd

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-96}"
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1

python -c "from vd_selectors import SolverType; print('solvers ok:', SolverType.AFS_Cox, SolverType.AFS_Multinomial)"
python -u "experiments/sec4_1_equivalence/sim_snr_${FAMILY}.py"
