#!/bin/bash
# Compare the judge base weights used in the run with the official gated google/gemma-2b and
# re-score a sample of completions with it. Needs Hugging Face access to google/gemma-2b.
#   sbatch --account=$SLURM_ACCOUNT -o $TAILRISK_ROOT/logs/%x-%j.out slurm/check_judge_base.sbatch
#SBATCH --job-name=tailrisk-basecheck
#SBATCH --partition=es0
#SBATCH --qos=es_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
module load "$TAILRISK_MODULE"
source "$TAILRISK_VENV/bin/activate"
python -u scripts/compare_judge_base.py --config "${CONFIG:-configs/experiment.yaml}" \
  --official google/gemma-2b --official-revision 9cf48e52b224239de00d483ec8eb84fb8d0f3a3a --rescore 512
