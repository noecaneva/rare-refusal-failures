#!/bin/bash
# Float16 batching sensitivity of the judge and distance from a float32 reference (one GPU).
#SBATCH --job-name=tailrisk-numerics
#SBATCH --partition=es0
#SBATCH --qos=es_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
module load "$TAILRISK_MODULE"
source "$TAILRISK_VENV/bin/activate"
python -u scripts/judge_numerics.py --config "${CONFIG:-configs/experiment.yaml}" --per-stratum 100 --cpu-per-stratum 30
