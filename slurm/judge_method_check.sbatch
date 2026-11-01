#!/bin/bash
# Pick an artifact-free judge attention path (one GPU).
#SBATCH --job-name=tailrisk-method
#SBATCH --partition=es0
#SBATCH --qos=es_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
module load "$TAILRISK_MODULE"
source "$TAILRISK_VENV/bin/activate"
python -u scripts/judge_method_check.py --config "${CONFIG:-configs/experiment.yaml}" --shard 5 --n-batches 16
