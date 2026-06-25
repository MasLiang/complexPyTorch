#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATADIR=${DATADIR:-"$ROOT_DIR/data"}
WORKDIR=${WORKDIR:-"$ROOT_DIR/bi_workdir"}
CONDA_ENV=${CONDA_ENV:-lut_net}
GPU_ID=${GPU_ID:-0}

LUT_SETS=${LUT_SETS:-1}
LUT_ALLOCATION=${LUT_ALLOCATION:-layer}
LUT_SETS_PER_CHANNEL=${LUT_SETS_PER_CHANNEL:-1}
NUM_EPOCHS=${NUM_EPOCHS:-400}
BATCH_SIZE=${BATCH_SIZE:-128}
START_FILTER=${START_FILTER:-11}
NUM_BLOCKS=${NUM_BLOCKS:-3}
OPTIMIZER=${OPTIMIZER:-sgd}

if [[ -n "${CHECKPOINT+x}" ]]; then
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
  fi
else
  CHECKPOINT="$WORKDIR/chkpts/Bestmodel_phase3.pt"
  if [[ ! -f "$CHECKPOINT" ]]; then
    LEGACY_CHECKPOINT="$WORKDIR/chkpts/Bestmodel.pt"
    if [[ -f "$LEGACY_CHECKPOINT" ]]; then
      CHECKPOINT="$LEGACY_CHECKPOINT"
      echo "Using legacy Phase 3 checkpoint: $CHECKPOINT"
    else
      echo "Missing Phase 3 checkpoint: $CHECKPOINT" >&2
      echo "Set WORKDIR to the directory containing chkpts/Bestmodel_phase3.pt, or pass CHECKPOINT=/path/to/checkpoint.pt." >&2
      exit 1
    fi
  fi
fi

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
  PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=(python)
fi

COMPILE_ARGS=()
if [[ "${COMPILE:-1}" == "1" ]]; then
  COMPILE_ARGS=(--compile --compile-backend "${COMPILE_BACKEND:-inductor}" --compile-mode "${COMPILE_MODE:-default}")
fi

echo "Running Phase 4"
echo "LUT allocation: $LUT_ALLOCATION"
echo "LUT sets per layer: $LUT_SETS"
echo "LUT sets per channel: $LUT_SETS_PER_CHANNEL"
echo "Checkpoint: $CHECKPOINT"
echo "Workdir: $WORKDIR"

"${PYTHON_CMD[@]}" "$ROOT_DIR/training.py" \
  --phase 4 \
  --datadir "$DATADIR" \
  --workdir "$WORKDIR" \
  --checkpoint "$CHECKPOINT" \
  --num-epochs "$NUM_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --start-filter "$START_FILTER" \
  --num-blocks "$NUM_BLOCKS" \
  --spectral-pool-scheme "${SPECTRAL_POOL_SCHEME:-none}" \
  --optimizer "$OPTIMIZER" \
  --lut-sets "$LUT_SETS" \
  --lut-allocation "$LUT_ALLOCATION" \
  --lut-sets-per-channel "$LUT_SETS_PER_CHANNEL" \
  "${COMPILE_ARGS[@]}" \
  "$@"
