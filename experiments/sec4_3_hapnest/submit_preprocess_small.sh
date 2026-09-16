#!/bin/bash
# UPSTREAM, EXTENDED: Slurm wrapper of taulantkoka/virtual-dummies (Koka et al., 2026),
# paths made configurable. GPLv3, see NOTICE.
#
# Preprocess all 100 replicates as a job array, one replicate per task:
#
#   bash prepare_hapnest_small.sh download && bash prepare_hapnest_small.sh arrange
#   mkdir -p logs && sbatch submit_preprocess_small.sh
#   bash prepare_hapnest_small.sh check
#
# Replace <project> and the paths below. RAW_DIR and DATA_DIR should sit on
# a scratch file system: the output is about 500 GB.
#
#SBATCH -J hapnest_preproc
#SBATCH -A <project>
#SBATCH --array=1-100
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o logs/preproc_%a_%j.out
#SBATCH -e logs/preproc_%a_%j.err

# activate the environment that has pandas-plink, e.g.
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate vd

export RAW_DIR="${RAW_DIR:-/path/to/scratch/hapnest_raw_small}"
export DATA_DIR="${DATA_DIR:-/path/to/scratch/hapnest_preprocessed_small}"
export OMP_NUM_THREADS=1

bash prepare_hapnest_small.sh preprocess
