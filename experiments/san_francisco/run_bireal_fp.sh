#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
cd "$ROOT_DIR"

exec "${PYTHON_BIN:-python}" -m experiments.san_francisco.train_bireal_flow \
  --stage bireal_fp \
  --data-root "${DATA_ROOT:-$ROOT_DIR/data/san_francisco}" \
  --workdir "${WORKDIR:-$ROOT_DIR/runs/san_francisco/bireal_fp}" \
  --patch-size "${PATCH_SIZE:-11}" \
  --split-mode "${SPLIT_MODE:-spatial}" \
  --spatial-block-size "${SPATIAL_BLOCK_SIZE:-64}" \
  --train-fraction "${TRAIN_FRACTION:-0.1}" \
  --val-fraction "${VAL_FRACTION:-0.1}" \
  --start-filters "${START_FILTERS:-16}" \
  --num-blocks "${NUM_BLOCKS:-3}" \
  --post-bn-mode "${POST_BN_MODE:-covariance}" \
  --shortcut-mode "${SHORTCUT_MODE:-fp}" \
  --epochs "${EPOCHS:-200}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-0.001}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --schedule "${SCHEDULE:-cosine}" \
  --min-lr-factor "${MIN_LR_FACTOR:-0.01}" \
  --seed "${SEED:-0}" \
  --split-seed "${SPLIT_SEED:-${SEED:-0}}" \
  --num-workers "${NUM_WORKERS:-4}" \
  "$@"
