#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATADIR=${DATADIR:-"$ROOT_DIR/data"}
WORKDIR=${WORKDIR:-"$ROOT_DIR/runs/phase3p5_lut5_qat"}
CHECKPOINT=${CHECKPOINT:-"$ROOT_DIR/bi_workdir/chkpts/Bestmodel_phase2.pt"}
CONDA_ENV=${CONDA_ENV:-lut_net}
GPU_ID=${GPU_ID:-0}

NUM_EPOCHS=${NUM_EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-128}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
TRANSITION_EPOCHS=${TRANSITION_EPOCHS:-100}
BETA_START=${BETA_START:-1.0}
BETA_END=${BETA_END:-8.0}

START_FILTER=${START_FILTER:-11}
NUM_BLOCKS=${NUM_BLOCKS:-3}
OPTIMIZER=${OPTIMIZER:-sgd}
LR=${LR:-0.001}
SCHEDULE=${SCHEDULE:-cosine}
MIN_LR_FACTOR=${MIN_LR_FACTOR:-0.1}
NO_VALIDATION=${NO_VALIDATION:-0}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing Phase 2 checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ "$SCHEDULE" != "constant" && "$SCHEDULE" != "cosine" ]]; then
  echo "Phase 3.5 requires SCHEDULE=constant or cosine" >&2
  exit 1
fi
if (( WARMUP_EPOCHS + TRANSITION_EPOCHS > NUM_EPOCHS )); then
  echo "WARMUP_EPOCHS + TRANSITION_EPOCHS must not exceed NUM_EPOCHS" >&2
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

echo "Running Phase 3.5 LUT5-aware activation QAT"
echo "Checkpoint: $CHECKPOINT"
echo "Workdir: $WORKDIR"
echo "QAT endpoint: Phase2 (1,1) -> hard LUT5 (2,0)"
echo "Warmup/transition/hard epochs: $WARMUP_EPOCHS/$TRANSITION_EPOCHS/$((NUM_EPOCHS - WARMUP_EPOCHS - TRANSITION_EPOCHS))"
echo "Comparator beta: $BETA_START -> $BETA_END"
echo "LR/schedule: $LR/$SCHEDULE"
echo "Compile: ${COMPILE:-0}"

"${PYTHON_CMD[@]}" "$ROOT_DIR/training.py" \
  --phase 3.5 \
  --lut-inputs 5 \
  --datadir "$DATADIR" \
  --workdir "$WORKDIR" \
  --checkpoint "$CHECKPOINT" \
  --num-epochs "$NUM_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --phase3p5-warmup-epochs "$WARMUP_EPOCHS" \
  --phase3p5-transition-epochs "$TRANSITION_EPOCHS" \
  --phase3p5-beta-start "$BETA_START" \
  --phase3p5-beta-end "$BETA_END" \
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
