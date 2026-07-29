#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PHASE2P1_MODE=${PHASE2P1_MODE:-c8}
DATADIR=${DATADIR:-"$ROOT_DIR/data"}
if [[ "$PHASE2P1_MODE" == "learned_lut5" ]]; then
  DEFAULT_WORKDIR="$ROOT_DIR/runs/phase2p1_learned_lut5"
else
  DEFAULT_WORKDIR="$ROOT_DIR/runs/phase2p1_c8"
fi
WORKDIR=${WORKDIR:-$DEFAULT_WORKDIR}
CHECKPOINT=${CHECKPOINT:-"$ROOT_DIR/bi_workdir/chkpts/Bestmodel_phase1.pt"}
CONDA_ENV=${CONDA_ENV:-lut_net}
GPU_ID=${GPU_ID:-0}

NUM_EPOCHS=${NUM_EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-128}
START_FILTER=${START_FILTER:-11}
NUM_BLOCKS=${NUM_BLOCKS:-3}
OPTIMIZER=${OPTIMIZER:-sgd}
LR=${LR:-0.001}
if [[ "$PHASE2P1_MODE" == "learned_lut5" ]]; then
  DEFAULT_SCHEDULE=cosine
else
  DEFAULT_SCHEDULE=default
fi
SCHEDULE=${SCHEDULE:-$DEFAULT_SCHEDULE}
MIN_LR_FACTOR=${MIN_LR_FACTOR:-0.1}
C8_BETA=${C8_BETA:-2.0}
C8_CODEBOOK=${C8_CODEBOOK:-roots}
C8_GRAD_MODE=${C8_GRAD_MODE:-softmax}
NO_VALIDATION=${NO_VALIDATION:-0}

LUT_SETS=${LUT_SETS:-1}
LUT_ALLOCATION=${LUT_ALLOCATION:-layer}
LUT_SETS_PER_CHANNEL=${LUT_SETS_PER_CHANNEL:-1}
LUT_INIT_MODE=${LUT_INIT_MODE:-raw}
LUT_LOGIT_INIT=${LUT_LOGIT_INIT:-2.0}
LUT_SIGN_FLIP_PROB=${LUT_SIGN_FLIP_PROB:-0}
LUT_LR=${LUT_LR:-0.01}
LUT_SCHEDULE=${LUT_SCHEDULE:-constant}
LUT_TAU_MIN=${LUT_TAU_MIN:-1.0}
LUT_TAU_MAX=${LUT_TAU_MAX:-10.0}
LUT_ANNEAL_EPOCHS=${LUT_ANNEAL_EPOCHS:-160}
LUT_HARD_TRANSITION_EPOCHS=${LUT_HARD_TRANSITION_EPOCHS:-20}
LUT_HARD_STE=${LUT_HARD_STE:-0}

if [[ "$PHASE2P1_MODE" != "c8" && "$PHASE2P1_MODE" != "learned_lut5" ]]; then
  echo "PHASE2P1_MODE must be c8 or learned_lut5" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing Phase 1 checkpoint: $CHECKPOINT" >&2
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

LUT_ARGS=()
if [[ "$PHASE2P1_MODE" == "learned_lut5" ]]; then
  LUT_ARGS=(
    --lut-sets "$LUT_SETS"
    --lut-allocation "$LUT_ALLOCATION"
    --lut-sets-per-channel "$LUT_SETS_PER_CHANNEL"
    --lut-init-mode "$LUT_INIT_MODE"
    --lut-logit-init "$LUT_LOGIT_INIT"
    --lut-sign-flip-prob "$LUT_SIGN_FLIP_PROB"
    --lut-lr "$LUT_LR"
    --lut-schedule "$LUT_SCHEDULE"
    --lut-tau-min "$LUT_TAU_MIN"
    --lut-tau-max "$LUT_TAU_MAX"
    --lut-anneal-epochs "$LUT_ANNEAL_EPOCHS"
    --lut-hard-transition-epochs "$LUT_HARD_TRANSITION_EPOCHS"
  )
  if [[ "$LUT_HARD_STE" == "1" ]]; then
    LUT_ARGS+=(--lut-hard-ste)
  fi
fi

echo "Running Phase 2.1 mode: $PHASE2P1_MODE"
echo "Checkpoint: $CHECKPOINT"
echo "Workdir: $WORKDIR"
if [[ "$PHASE2P1_MODE" == "learned_lut5" ]]; then
  echo "Operator: differentiable LUT5 [sr,si,d,wr,wi] -> [br,bi]"
  echo "Initialization: duplicated LUT4 $LUT_INIT_MODE table, logit=$LUT_LOGIT_INIT"
  echo "LUT LR/schedule: $LUT_LR/$LUT_SCHEDULE"
  echo "LUT tau/anneal/transition: $LUT_TAU_MIN->$LUT_TAU_MAX / $LUT_ANNEAL_EPOCHS / $LUT_HARD_TRANSITION_EPOCHS"
  echo "Hard checkpoint: only after LUT hard transition completes"
else
  echo "Operator: C8 activation + binary complex weight + ordinary complex convolution"
  echo "Local >=0 before accumulation: disabled"
  echo "C8 codebook: $C8_CODEBOOK"
  echo "C8 gradient mode: $C8_GRAD_MODE"
  echo "C8 backward beta: $C8_BETA"
fi
echo "LR/schedule: $LR/$SCHEDULE"
echo "Compile: ${COMPILE:-0}"

"${PYTHON_CMD[@]}" "$ROOT_DIR/training.py" \
  --phase 2.1 \
  --phase2p1-mode "$PHASE2P1_MODE" \
  --lut-inputs 5 \
  --c8-beta "$C8_BETA" \
  --c8-codebook "$C8_CODEBOOK" \
  --c8-grad-mode "$C8_GRAD_MODE" \
  --datadir "$DATADIR" \
  --workdir "$WORKDIR" \
  --checkpoint "$CHECKPOINT" \
  --num-epochs "$NUM_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --start-filter "$START_FILTER" \
  --num-blocks "$NUM_BLOCKS" \
  --spectral-pool-scheme "${SPECTRAL_POOL_SCHEME:-none}" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --schedule "$SCHEDULE" \
  --min-lr-factor "$MIN_LR_FACTOR" \
  "${LUT_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  "$@"
