#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHECKPOINT=${CHECKPOINT:-"$ROOT_DIR/runs/san_francisco/bireal_lut/best.pt"}
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
cd "$ROOT_DIR"

exec "${PYTHON_BIN:-python}" -m experiments.san_francisco.train_bireal_flow \
  --stage lut6_residual \
  --checkpoint "$CHECKPOINT" \
  --data-root "${DATA_ROOT:-$ROOT_DIR/data/san_francisco}" \
  --workdir "${WORKDIR:-$ROOT_DIR/runs/san_francisco/lut6_residual}" \
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
  --lr "${LR:-0.0002}" \
  --lut-lr "${LUT_LR:-0.002}" \
  --base-lut-lr 0 \
  --residual-alpha-start "${RESIDUAL_ALPHA_START:-0.1}" \
  --residual-alpha-end "${RESIDUAL_ALPHA_END:-1.0}" \
  --residual-ramp-epochs "${RESIDUAL_RAMP_EPOCHS:-120}" \
  --base-alpha-start 0 \
  --base-alpha-end 0 \
  --base-ramp-epochs 1 \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --schedule "${SCHEDULE:-constant}" \
  --seed "${SEED:-0}" \
  --split-seed "${SPLIT_SEED:-${SEED:-0}}" \
  --num-workers "${NUM_WORKERS:-4}" \
  "$@"
