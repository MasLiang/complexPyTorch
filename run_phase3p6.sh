#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATADIR=${DATADIR:-"$ROOT_DIR/data"}
WORKDIR=${WORKDIR:-"$ROOT_DIR/runs/phase3p6_c8_cosine"}
CHECKPOINT=${CHECKPOINT:-"$ROOT_DIR/bi_workdir/chkpts/Bestmodel_phase3.pt"}
CONDA_ENV=${CONDA_ENV:-lut_net}
GPU_ID=${GPU_ID:-0}

NUM_EPOCHS=${NUM_EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-128}
WARMUP_EPOCHS=${C8_WARMUP_EPOCHS:-10}
TRANSITION_EPOCHS=${C8_TRANSITION_EPOCHS:-100}
BETA_START=${C8_BETA_START:-2.0}
BETA_END=${C8_BETA_END:-12.0}
CODE_SCHEDULE=${C8_SCHEDULE:-cosine}

START_FILTER=${START_FILTER:-11}
NUM_BLOCKS=${NUM_BLOCKS:-3}
OPTIMIZER=${OPTIMIZER:-sgd}
LR=${LR:-0.001}
SCHEDULE=${SCHEDULE:-cosine}
MIN_LR_FACTOR=${MIN_LR_FACTOR:-0.1}
NO_VALIDATION=${NO_VALIDATION:-0}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing LUT4 Phase 3 checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ "$SCHEDULE" != "constant" && "$SCHEDULE" != "cosine" ]]; then
  echo "Phase 3.6 requires SCHEDULE=constant or cosine" >&2
  exit 1
fi
if [[ "$CODE_SCHEDULE" != "linear" && "$CODE_SCHEDULE" != "cosine" ]]; then
  echo "C8_SCHEDULE must be linear or cosine" >&2
  exit 1
fi
if (( WARMUP_EPOCHS + TRANSITION_EPOCHS > NUM_EPOCHS )); then
  echo "C8_WARMUP_EPOCHS + C8_TRANSITION_EPOCHS must not exceed NUM_EPOCHS" >&2
  exit 1
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

DATA_ARGS=()
if [[ "$NO_VALIDATION" == "1" ]]; then
  DATA_ARGS+=(--no-validation)
fi

COMPILE_ARGS=()
if [[ "${COMPILE:-0}" == "1" ]]; then
  COMPILE_ARGS=(
    --compile
    --compile-backend "${COMPILE_BACKEND:-aot_eager}"
    --compile-mode "${COMPILE_MODE:-default}"
  )
fi

echo "Running Phase 3.6 analytic nearest-C8 LUT-aware QAT"
echo "Checkpoint: $CHECKPOINT"
echo "Workdir: $WORKDIR"
echo "Future physical mapping: 3-bit X phase + 2-bit complex W -> real/imag LUT5"
echo "Training operation: analytic decode + fixed complex multiply + local >=0 (no LUT backend)"
echo "Warmup/transition/full-C8 epochs: $WARMUP_EPOCHS/$TRANSITION_EPOCHS/$((NUM_EPOCHS - WARMUP_EPOCHS - TRANSITION_EPOCHS))"
echo "C8 schedule: $CODE_SCHEDULE"
echo "C8 backward beta: $BETA_START -> $BETA_END"
echo "LR/schedule: $LR/$SCHEDULE"
echo "Compile: ${COMPILE:-0}"

"${PYTHON_CMD[@]}" "$ROOT_DIR/training.py" \
  --phase 3.6 \
  --lut-inputs 5 \
  --datadir "$DATADIR" \
  --workdir "$WORKDIR" \
  --checkpoint "$CHECKPOINT" \
  --num-epochs "$NUM_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --phase3p6-warmup-epochs "$WARMUP_EPOCHS" \
  --phase3p6-transition-epochs "$TRANSITION_EPOCHS" \
  --phase3p6-beta-start "$BETA_START" \
  --phase3p6-beta-end "$BETA_END" \
  --phase3p6-schedule "$CODE_SCHEDULE" \
  --start-filter "$START_FILTER" \
  --num-blocks "$NUM_BLOCKS" \
  --spectral-pool-scheme "${SPECTRAL_POOL_SCHEME:-none}" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --schedule "$SCHEDULE" \
  --min-lr-factor "$MIN_LR_FACTOR" \
  "${DATA_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  "$@"
