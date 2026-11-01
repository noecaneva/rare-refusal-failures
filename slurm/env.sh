#!/bin/bash
# Source this before any work. Everything heavy (venv, model caches, completions, logs)
# lives under $TAILRISK_ROOT on scratch; the repo holds only code, configs and aggregates.
export TAILRISK_ROOT=${TAILRISK_ROOT:-$SCRATCH/qwen_tailrisk}
export HF_HOME=$TAILRISK_ROOT/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_DISABLE_TELEMETRY=1
# A Hugging Face token (only needed for the gated judge base google/gemma-2b) stays in the home
# directory, not in the scratch cache: `hf auth login` with HF_HOME unset writes it here.
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export PIP_CACHE_DIR=$TAILRISK_ROOT/pip_cache
export XDG_CACHE_HOME=$TAILRISK_ROOT/xdg_cache
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TORCH_HOME=$XDG_CACHE_HOME/torch
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/inductor
export TMPDIR=$TAILRISK_ROOT/tmp
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$VLLM_CACHE_ROOT" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$TAILRISK_ROOT"/{data/completions,data/scores,logs,envs}
# Cluster software stack: the vLLM module provides torch, vllm and transformers; the venv
# (created by scripts/setup_env.sh with --system-site-packages) adds the rest.
export TAILRISK_MODULE=${TAILRISK_MODULE:-ai/vllm/0.19.0}
export TAILRISK_VENV=$TAILRISK_ROOT/envs/venv
