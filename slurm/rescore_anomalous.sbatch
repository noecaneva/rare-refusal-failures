#!/bin/bash
# Re-score the artifact-affected judge outputs in place, one es0 node, one worker per GPU
# (shards split round-robin). RESCORE_ARGS passes --attn / --batch-size / --verify.
#SBATCH --job-name=tailrisk-rescore
#SBATCH --partition=es0
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
module load "$TAILRISK_MODULE"
source "$TAILRISK_VENV/bin/activate"
CONFIG=${CONFIG:-configs/experiment.yaml}
N=$(python -c "import sys; sys.path.insert(0,'src'); from common import load_config, n_shards; print(n_shards(load_config('$CONFIG')))")
NGPU=$(nvidia-smi -L | wc -l)
PIDS=()
for ((g = 0; g < NGPU; g++)); do
  SH=$(seq -s, $g $NGPU $((N - 1)))
  CUDA_VISIBLE_DEVICES=$g python -u scripts/rescore_anomalous.py --config "$CONFIG" --shards "$SH" ${RESCORE_ARGS:-} \
    > "$TAILRISK_ROOT/logs/tailrisk-rescore-$SLURM_JOB_ID-gpu$g.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0; for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
cat "$TAILRISK_ROOT"/logs/tailrisk-rescore-$SLURM_JOB_ID-gpu*.log | grep "^shard"
exit $FAIL
