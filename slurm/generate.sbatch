#!/bin/bash
# Stage 1: sample completions. Each array task takes a whole es0 node (4 GPUs) and runs one
# worker per GPU; workers claim shards from a lock-file queue until none are left, so GPUs stay
# busy and nothing is generated twice. Resubmitting continues where a killed run stopped.
# Submit through scripts/submit.sh, which builds the array from the shards still missing and
# passes --account / -o. The account is deliberately not hard-coded here: set SLURM_ACCOUNT
# (see slurm/local.sh.example) or export SBATCH_ACCOUNT before calling sbatch directly.
#SBATCH --job-name=tailrisk-gen
#SBATCH --partition=es0
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
module load "$TAILRISK_MODULE"
source "$TAILRISK_VENV/bin/activate"
CONFIG=${CONFIG:-configs/experiment.yaml}
echo "job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname -s | sed 's/[0-9]*//g') $(date -u +%FT%TZ) config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
slurm/workers.sh generate "$CONFIG"
