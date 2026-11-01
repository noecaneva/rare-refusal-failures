#!/bin/bash
# Stage 2: score completions with the StrongREJECT judge. Same node/worker layout as generate.
# Submit through scripts/submit.sh (see slurm/generate.sbatch for the account convention).
#SBATCH --job-name=tailrisk-judge
#SBATCH --partition=es0
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
module load "$TAILRISK_MODULE"
source "$TAILRISK_VENV/bin/activate"
CONFIG=${CONFIG:-configs/experiment.yaml}
echo "job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID:-0} $(date -u +%FT%TZ) config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
slurm/workers.sh judge "$CONFIG" ${JUDGE_ARGS:-}
