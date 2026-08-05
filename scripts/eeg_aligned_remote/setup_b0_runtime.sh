#!/usr/bin/env bash
set -euo pipefail

ROOT="${B0_ROOT:-/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned}"
RUNTIME_DIR="$ROOT/runtime"
ENV_PREFIX="${B0_ENV:-$RUNTIME_DIR/envs/b0}"
MAMBA="$RUNTIME_DIR/bin/micromamba"

mkdir -p "$RUNTIME_DIR"

if [ ! -x "$MAMBA" ]; then
  cd "$RUNTIME_DIR"
  curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest -o micromamba.tar.bz2
  tar -xjf micromamba.tar.bz2 bin/micromamba
fi

export MAMBA_ROOT_PREFIX="$RUNTIME_DIR/mamba_root"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  "$MAMBA" create -y -p "$ENV_PREFIX" \
    -c pytorch -c conda-forge \
    python=3.11 numpy scipy scikit-learn pandas tqdm pytorch cpuonly
fi

"$ENV_PREFIX/bin/python" - <<'PY'
import numpy
import scipy
import sklearn
import torch

print("python_ok")
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "$ENV_PREFIX/bin/python"
