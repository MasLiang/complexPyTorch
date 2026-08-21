#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export GPU_ID=${GPU_ID:-0}
export DATADIR=${DATADIR:-"$ROOT_DIR/data"}
export WORKDIR=${WORKDIR:-"$ROOT_DIR/runs/phase2_complex_bireal"}

NUM_EPOCHS=${NUM_EPOCHS:-256}
BATCH_SIZE=${BATCH_SIZE:-128}
START_FILTER=${START_FILTER:-16}
NUM_BLOCKS=${NUM_BLOCKS:-3}
LR=${LR:-0.01}
SCHEDULE=${SCHEDULE:-bireal_reference}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
POST_BN_MODE=${POST_BN_MODE:-covariance}
NUM_WORKERS=${NUM_WORKERS:-8}

ARGS=(
  --phase 2
  --num-epochs "$NUM_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --start-filter "$START_FILTER"
  --num-blocks "$NUM_BLOCKS"
  --optimizer adam
  --lr "$LR"
  --schedule "$SCHEDULE"
  --weight-decay "$WEIGHT_DECAY"
  --clipnorm 0
  --clipval 0
  --spectral-pool-scheme none
  --binary-weight-scale channel
  --weight-grad-mode ste
  --activation-grad-mode bireal
  --post-bn-mode "$POST_BN_MODE"
  --augmentation real_lut
  --label-smoothing 0.1
  --no-validation
  --num-workers "$NUM_WORKERS"
)

if [[ -n "${CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
else
  ARGS+=(--train-from-scratch)
fi

printf 'Running complex Bi-Real Phase 2 on GPU %s\n' "$GPU_ID"
printf 'Workdir: %s\n' "$WORKDIR"
exec "$ROOT_DIR/run_training.sh" "${ARGS[@]}" "$@"
