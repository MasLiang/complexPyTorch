#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export GPU_ID=${GPU_ID:-0}
export DATADIR=${DATADIR:-"$ROOT_DIR/data"}
export WORKDIR=${WORKDIR:-"$ROOT_DIR/runs/phase2"}

NUM_EPOCHS=${NUM_EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-128}
START_FILTER=${START_FILTER:-11}
NUM_BLOCKS=${NUM_BLOCKS:-3}
LR=${LR:-0.1}
SCHEDULE=${SCHEDULE:-bireal}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
SPECTRAL_POOL_SCHEME=${SPECTRAL_POOL_SCHEME:-none}
BINARY_WEIGHT_SCALE=${BINARY_WEIGHT_SCALE:-channel}

ARGS=(
  --phase 2
  --num-epochs "$NUM_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --start-filter "$START_FILTER"
  --num-blocks "$NUM_BLOCKS"
  --lr "$LR"
  --schedule "$SCHEDULE"
  --weight-decay "$WEIGHT_DECAY"
  --spectral-pool-scheme "$SPECTRAL_POOL_SCHEME"
  --binary-weight-scale "$BINARY_WEIGHT_SCALE"
)

if [[ "${TRAIN_FROM_SCRATCH:-0}" == "1" ]]; then
  ARGS+=(--train-from-scratch)
else
  CHECKPOINT=${CHECKPOINT:-"$ROOT_DIR/bi_workdir/chkpts/Bestmodel_phase1.pt"}
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

printf 'Running Phase 2 on GPU %s\n' "$GPU_ID"
printf 'Workdir: %s\n' "$WORKDIR"
if [[ "${TRAIN_FROM_SCRATCH:-0}" != "1" ]]; then
  printf 'Checkpoint: %s\n' "$CHECKPOINT"
fi
exec "$ROOT_DIR/run_training.sh" "${ARGS[@]}" "$@"
