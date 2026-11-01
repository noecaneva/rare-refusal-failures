#!/bin/bash
# Launch one worker per visible GPU, each claiming shards from the queue; wait for all of them
# and fail if any failed. Usage: slurm/workers.sh <stage: generate|judge> <config> [extra args]
STAGE=$1; CONFIG=$2; shift 2
NGPU=$(nvidia-smi -L | wc -l)
LOGDIR=$TAILRISK_ROOT/logs
PIDS=()
for ((g = 0; g < NGPU; g++)); do
  # Per-worker compile caches: concurrent workers writing one inductor/Triton cache corrupt it.
  export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/inductor/gpu$g TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton/gpu$g
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  CUDA_VISIBLE_DEVICES=$g python -u "src/$STAGE.py" --config "$CONFIG" --claim \
    --owner "${SLURM_JOB_ID:-local}:$g" "$@" > "$LOGDIR/${SLURM_JOB_NAME:-$STAGE}-${SLURM_JOB_ID:-local}-gpu$g.log" 2>&1 &
  PIDS+=($!)
  sleep 10
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
for ((g = 0; g < NGPU; g++)); do echo "--- worker gpu$g (last lines)"; tail -n 4 "$LOGDIR/${SLURM_JOB_NAME:-$STAGE}-${SLURM_JOB_ID:-local}-gpu$g.log"; done
exit $FAIL
