#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DATADIR=${DATADIR:-"."}
WORKDIR=${WORKDIR:-"."}
GPU_ID=${GPU_ID:-}
if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

COMPILE_ARGS=()
if [[ "${COMPILE:-0}" == "1" ]]; then
  COMPILE_ARGS=(--compile --compile-backend "${COMPILE_BACKEND:-inductor}" --compile-mode "${COMPILE_MODE:-default}")
fi

NUM_GPUS=${NUM_GPUS:-1}

EXTRA_ARGS=("$@")

if [[ "${USE_DDP:-0}" == "1" ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node="$NUM_GPUS" \
      "$ROOT_DIR/training.py" \
      --ddp \
      --datadir "$DATADIR" \
      --workdir "$WORKDIR" \
      "${COMPILE_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  else
    "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NUM_GPUS" \
      "$ROOT_DIR/training.py" \
      --ddp \
      --datadir "$DATADIR" \
      --workdir "$WORKDIR" \
      "${COMPILE_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  fi
else
  "$PYTHON_BIN" "$ROOT_DIR/training.py" \
    --datadir "$DATADIR" \
    --workdir "$WORKDIR" \
    "${COMPILE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
fi
