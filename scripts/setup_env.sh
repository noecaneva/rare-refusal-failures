#!/bin/bash
# One-time environment setup (safe on a login node: no GPU work, only pip).
#   source slurm/env.sh && bash scripts/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/../slurm/env.sh"
module load "$TAILRISK_MODULE"
if [ ! -x "$TAILRISK_VENV/bin/python" ]; then
  python -m venv --system-site-packages "$TAILRISK_VENV"
fi
source "$TAILRISK_VENV/bin/activate"
# Freeze the module's packages as constraints so pip cannot replace the module's torch/vllm/
# transformers with incompatible versions.
pip freeze --exclude-editable 2>/dev/null | grep -E "^[A-Za-z0-9_.-]+==" > "$TMPDIR/module_constraints.txt" || true
pip install --upgrade pip
pip install -c "$TMPDIR/module_constraints.txt" -r "$(dirname "$0")/../requirements.txt"
python -c "import vllm, transformers, torch, strong_reject; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
