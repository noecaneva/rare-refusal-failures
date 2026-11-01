#!/bin/bash
# Submit a stage. Full-run stages are job arrays of node jobs (4 GPUs, 4 workers each) that
# pull shards from a queue, so the array size is "how many nodes at once", not the shard count.
#   scripts/submit.sh pilot [configs/pilot.yaml|configs/smoke.yaml]
#   scripts/submit.sh generate [config] [n_node_jobs] [after_job_id]
#   scripts/submit.sh judge    [config] [n_node_jobs] [after_job_id]
# Account / QoS come from the environment: SLURM_ACCOUNT (required) and optionally SBATCH_QOS,
# e.g. from an untracked slurm/local.sh (see slurm/local.sh.example).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f slurm/local.sh ] && source slurm/local.sh
source slurm/env.sh
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT (see slurm/local.sh.example)}"
STAGE=$1
LOGDIR=$TAILRISK_ROOT/logs
mkdir -p "$LOGDIR"
if [ "$STAGE" = pilot ]; then
  exec sbatch --account="$SLURM_ACCOUNT" -o "$LOGDIR/%x-%j.out" --export=ALL,CONFIG="${2:-configs/pilot.yaml}" slurm/pilot.sbatch
fi
CONFIG=${2:-configs/experiment.yaml}
NJOBS=${3:-2}
AFTER=${4:-}
module load "$TAILRISK_MODULE" >/dev/null 2>&1
source "$TAILRISK_VENV/bin/activate"
python src/shards.py --config "$CONFIG" --clean-locks
MISSING=$(python src/shards.py --config "$CONFIG" --stage "$STAGE")
if [ -z "$MISSING" ] && [ -z "$AFTER" ]; then echo "nothing to do for $STAGE ($CONFIG)"; exit 0; fi
N=$(echo "$MISSING" | tr ',' '\n' | grep -c . || true)
echo "$STAGE: $N shards pending now; submitting $NJOBS node job(s)${AFTER:+ after job $AFTER}"
DEP=(); [ -n "$AFTER" ] && DEP=(--dependency="afterany:$AFTER")
exec sbatch --account="$SLURM_ACCOUNT" -o "$LOGDIR/%x-%A_%a.out" --array="0-$((NJOBS - 1))" "${DEP[@]}" \
     --export=ALL,CONFIG="$CONFIG" "slurm/$STAGE.sbatch"
