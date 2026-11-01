#!/bin/bash
# Pilot (or smoke) run: generation + judging (with an fp32 self-check) + analysis in one job on
# one es0 node, under the short debug QoS. One worker per GPU; the log reports wall-clock,
# throughput and GPU memory per stage.
#   scripts/submit.sh pilot                      # configs/pilot.yaml: 10 prompts x 1,000 samples
#   scripts/submit.sh pilot configs/smoke.yaml   # 2 prompts x 40 samples, a few minutes
#SBATCH --job-name=tailrisk-pilot
#SBATCH --partition=es0
#SBATCH --qos=es_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
module load "$TAILRISK_MODULE"
source "$TAILRISK_VENV/bin/activate"
CONFIG=${CONFIG:-configs/pilot.yaml}
NAME=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['name'])")
echo "pilot job $SLURM_JOB_ID $(date -u +%FT%TZ) config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SECONDS=0
slurm/workers.sh generate "$CONFIG"
echo "generation wall-clock: ${SECONDS}s"
SECONDS=0
slurm/workers.sh judge "$CONFIG" --selfcheck 64
echo "judging wall-clock: ${SECONDS}s"
python -u src/analyze.py --config "$CONFIG" --out "results/$NAME"
python src/shards.py --config "$CONFIG" --status
