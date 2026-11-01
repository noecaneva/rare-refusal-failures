#!/bin/bash
# Download the paper's Llama-3.1-8B per-prompt scores and build the reference curve (light work, but the es0 QoS requires one GPU per job).
#SBATCH --job-name=tailrisk-ref
#SBATCH --partition=es0
#SBATCH --qos=es_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
module load "$TAILRISK_MODULE"
source "$TAILRISK_VENV/bin/activate"
python -u scripts/reference_from_paper.py --config "${CONFIG:-configs/experiment.yaml}"
