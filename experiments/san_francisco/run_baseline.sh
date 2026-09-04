#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_ID=${GPU_ID:-0}
DATA_ROOT=${DATA_ROOT:-"$ROOT_DIR/data/san_francisco"}
WORKDIR=${WORKDIR:-"$ROOT_DIR/runs/san_francisco/fp_baseline"}
PYTHON_BIN=${PYTHON_BIN:-python}

export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd "$ROOT_DIR"

exec "$PYTHON_BIN" -m experiments.san_francisco.train \
  --data-root "$DATA_ROOT" \
  --workdir "$WORKDIR" \
  --conv-type "${CONV_TYPE:-fp}" \
  --patch-size "${PATCH_SIZE:-11}" \
  --split-mode "${SPLIT_MODE:-spatial}" \
  --spatial-block-size "${SPATIAL_BLOCK_SIZE:-64}" \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-0.001}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --seed "${SEED:-0}" \
  --split-seed "${SPLIT_SEED:-${SEED:-0}}" \
  --train-fraction "${TRAIN_FRACTION:-0.1}" \
  --val-fraction "${VAL_FRACTION:-0.1}" \
  --num-workers "${NUM_WORKERS:-4}" \
  "$@"
