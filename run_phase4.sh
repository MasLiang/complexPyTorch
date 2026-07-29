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
LUT_INPUTS=${LUT_INPUTS:-4}
LUT_EXTRA_BIT=${LUT_EXTRA_BIT:-phase}
LUT_LOGIT_INIT=${LUT_LOGIT_INIT:-5.0}
LUT_INIT_MODE=${LUT_INIT_MODE:-binary}
LUT_SIGN_FLIP_PROB=${LUT_SIGN_FLIP_PROB:-0.0}
LUT_LR=${LUT_LR:-0.001}
LUT_SCHEDULE=${LUT_SCHEDULE:-follow_base}
LUT_TAU_MIN=${LUT_TAU_MIN:-1.0}
LUT_TAU_MAX=${LUT_TAU_MAX:-10.0}
LUT_ANNEAL_EPOCHS=${LUT_ANNEAL_EPOCHS:-}
LUT_HARD_STE=${LUT_HARD_STE:-0}
LUT_HARD_EPOCH_FRACTION=${LUT_HARD_EPOCH_FRACTION:-0.9}
LUT_HARD_TRANSITION_EPOCHS=${LUT_HARD_TRANSITION_EPOCHS:-0}
LUT_SOFT_ONLY_EPOCHS=${LUT_SOFT_ONLY_EPOCHS:-0}
LUT_HARD_BN_ONLY_EPOCHS=${LUT_HARD_BN_ONLY_EPOCHS:-0}
LUT_SEARCH_EPOCHS=${LUT_SEARCH_EPOCHS:-0}
LUT_SEARCH_WEIGHT_EPOCHS_PER_LUT=${LUT_SEARCH_WEIGHT_EPOCHS_PER_LUT:-1}
LUT_SEARCH_MAX_FLIPS_PER_COMMIT=${LUT_SEARCH_MAX_FLIPS_PER_COMMIT:-8}
LUT_SEARCH_MAX_FLIPS_PER_LAYER=${LUT_SEARCH_MAX_FLIPS_PER_LAYER:-1}
LUT_SEARCH_FLIP_COOLDOWN=${LUT_SEARCH_FLIP_COOLDOWN:-2}
LUT_SEARCH_INIT_FLIP_PROB=${LUT_SEARCH_INIT_FLIP_PROB:-0.05}
LUT_SEARCH_SCORE_TEMPERATURE=${LUT_SEARCH_SCORE_TEMPERATURE:-0.25}
LUT_SEARCH_COMMIT_THRESHOLD=${LUT_SEARCH_COMMIT_THRESHOLD:-0.8}
LUT_SEARCH_SPARSITY=${LUT_SEARCH_SPARSITY:-0.0}
LUT_SEARCH_FREEZE_BN=${LUT_SEARCH_FREEZE_BN:-1}
NUM_EPOCHS=${NUM_EPOCHS:-400}
BATCH_SIZE=${BATCH_SIZE:-128}
START_FILTER=${START_FILTER:-11}
NUM_BLOCKS=${NUM_BLOCKS:-3}
OPTIMIZER=${OPTIMIZER:-sgd}
LR=${LR:-0.001}
SCHEDULE=${SCHEDULE:-default}
MIN_LR_FACTOR=${MIN_LR_FACTOR:-0.1}
NO_VALIDATION=${NO_VALIDATION:-0}
TRAIN_FROM_SCRATCH=${TRAIN_FROM_SCRATCH:-0}
COMP_INIT=${COMP_INIT:-complex_independent}
MAGNITUDE_THRESHOLD_INIT=${MAGNITUDE_THRESHOLD_INIT:-1.0}
MAGNITUDE_THRESHOLD_LR=${MAGNITUDE_THRESHOLD_LR:-0.001}
MAGNITUDE_THRESHOLD_SCHEDULE=${MAGNITUDE_THRESHOLD_SCHEDULE:-follow_base}
MAGNITUDE_BIT_BETA=${MAGNITUDE_BIT_BETA:-2.0}
MAGNITUDE_SHADOW_EPSILON=${MAGNITUDE_SHADOW_EPSILON:-0.05}

CHECKPOINT_ARGS=()
if [[ "$TRAIN_FROM_SCRATCH" == "1" ]]; then
  CHECKPOINT="<none: train from scratch>"
  CHECKPOINT_ARGS=(--train-from-scratch)
elif [[ -n "${CHECKPOINT+x}" ]]; then
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
  fi
  CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
else
  if [[ "$LUT_EXTRA_BIT" == "magnitude_ste" ]]; then
    CHECKPOINT="$WORKDIR/chkpts/Bestmodel_phase3.pt"
  elif [[ "$LUT_INPUTS" == "5" && -f "$WORKDIR/chkpts/Bestmodel_phase3p5.pt" ]]; then
    CHECKPOINT="$WORKDIR/chkpts/Bestmodel_phase3p5.pt"
    echo "Using Phase 3.5 LUT5-aware QAT checkpoint: $CHECKPOINT"
  else
    CHECKPOINT="$WORKDIR/chkpts/Bestmodel_phase3.pt"
  fi
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
  CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
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

ANNEAL_ARGS=()
if [[ -n "$LUT_ANNEAL_EPOCHS" ]]; then
  ANNEAL_ARGS+=(--lut-anneal-epochs "$LUT_ANNEAL_EPOCHS")
fi
if [[ "$LUT_HARD_STE" == "1" ]]; then
  ANNEAL_ARGS+=(--lut-hard-ste)
fi

LUT_SEARCH_ARGS=()
if [[ "$LUT_SEARCH_EPOCHS" -gt 0 ]]; then
  LUT_SEARCH_ARGS+=(
    --lut-search-epochs "$LUT_SEARCH_EPOCHS"
    --lut-search-weight-epochs-per-lut "$LUT_SEARCH_WEIGHT_EPOCHS_PER_LUT"
    --lut-search-max-flips-per-commit "$LUT_SEARCH_MAX_FLIPS_PER_COMMIT"
    --lut-search-max-flips-per-layer "$LUT_SEARCH_MAX_FLIPS_PER_LAYER"
    --lut-search-flip-cooldown "$LUT_SEARCH_FLIP_COOLDOWN"
    --lut-search-init-flip-prob "$LUT_SEARCH_INIT_FLIP_PROB"
    --lut-search-score-temperature "$LUT_SEARCH_SCORE_TEMPERATURE"
    --lut-search-commit-threshold "$LUT_SEARCH_COMMIT_THRESHOLD"
    --lut-search-sparsity "$LUT_SEARCH_SPARSITY"
  )
  if [[ "$LUT_SEARCH_FREEZE_BN" == "1" ]]; then
    LUT_SEARCH_ARGS+=(--lut-search-freeze-bn)
  fi
fi

DATA_ARGS=()
if [[ "$NO_VALIDATION" == "1" ]]; then
  DATA_ARGS+=(--no-validation)
fi

echo "Running Phase 4"
echo "LUT inputs: $LUT_INPUTS"
if [[ "$LUT_INPUTS" == "5" ]]; then
  if [[ "$LUT_EXTRA_BIT" == "magnitude_ste" ]]; then
    echo "LUT5 magnitude bit: 1[abs(BN_pre(x)) >= theta_c]"
    echo "Magnitude threshold init/lr/schedule: $MAGNITUDE_THRESHOLD_INIT / $MAGNITUDE_THRESHOLD_LR / $MAGNITUDE_THRESHOLD_SCHEDULE"
    echo "Magnitude backward beta/shadow epsilon: $MAGNITUDE_BIT_BETA / $MAGNITUDE_SHADOW_EPSILON"
  else
    echo "LUT5 phase bit: 1[abs(x_r) >= abs(x_i)]"
  fi
fi
echo "LUT logit init: $LUT_LOGIT_INIT"
echo "LUT init mode: $LUT_INIT_MODE"
echo "LUT sign flip prob: $LUT_SIGN_FLIP_PROB"
echo "LUT lr: $LUT_LR"
echo "LUT schedule: $LUT_SCHEDULE"
echo "LUT tau: $LUT_TAU_MIN -> $LUT_TAU_MAX"
if [[ -n "$LUT_ANNEAL_EPOCHS" ]]; then
  echo "LUT anneal epochs: $LUT_ANNEAL_EPOCHS"
fi
echo "LUT hard STE: $LUT_HARD_STE"
echo "LUT hard epoch fraction: $LUT_HARD_EPOCH_FRACTION"
echo "LUT hard transition epochs: $LUT_HARD_TRANSITION_EPOCHS"
echo "LUT soft-only epochs: $LUT_SOFT_ONLY_EPOCHS"
echo "LUT hard+BN-only epochs: $LUT_HARD_BN_ONLY_EPOCHS"
echo "LUT alternating search epochs: $LUT_SEARCH_EPOCHS"
if [[ "$LUT_SEARCH_EPOCHS" -gt 0 ]]; then
  echo "Weight epochs per LUT epoch: $LUT_SEARCH_WEIGHT_EPOCHS_PER_LUT"
  echo "LUT max flips per commit: $LUT_SEARCH_MAX_FLIPS_PER_COMMIT"
  echo "LUT max flips per layer: $LUT_SEARCH_MAX_FLIPS_PER_LAYER"
  echo "LUT flip cooldown: $LUT_SEARCH_FLIP_COOLDOWN"
  echo "LUT search init flip prob: $LUT_SEARCH_INIT_FLIP_PROB"
  echo "LUT search score temperature: $LUT_SEARCH_SCORE_TEMPERATURE"
  echo "LUT search commit threshold: $LUT_SEARCH_COMMIT_THRESHOLD"
  echo "LUT search sparsity: $LUT_SEARCH_SPARSITY"
  echo "LUT search freeze BN: $LUT_SEARCH_FREEZE_BN"
fi
echo "LUT allocation: $LUT_ALLOCATION"
echo "LUT sets per layer: $LUT_SETS"
echo "LUT sets per channel: $LUT_SETS_PER_CHANNEL"
echo "Train from scratch: $TRAIN_FROM_SCRATCH"
echo "Checkpoint: $CHECKPOINT"
echo "Base lr: $LR"
echo "Schedule: $SCHEDULE"
echo "Min LR factor: $MIN_LR_FACTOR"
echo "No validation: $NO_VALIDATION"
echo "Complex init: $COMP_INIT"
echo "Workdir: $WORKDIR"

"${PYTHON_CMD[@]}" "$ROOT_DIR/training.py" \
  --phase 4 \
  --datadir "$DATADIR" \
  --workdir "$WORKDIR" \
  "${CHECKPOINT_ARGS[@]}" \
  --num-epochs "$NUM_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --start-filter "$START_FILTER" \
  --num-blocks "$NUM_BLOCKS" \
  --spectral-pool-scheme "${SPECTRAL_POOL_SCHEME:-none}" \
  --comp_init "$COMP_INIT" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --schedule "$SCHEDULE" \
  --min-lr-factor "$MIN_LR_FACTOR" \
  "${DATA_ARGS[@]}" \
  --lut-sets "$LUT_SETS" \
  --lut-allocation "$LUT_ALLOCATION" \
  --lut-sets-per-channel "$LUT_SETS_PER_CHANNEL" \
  --lut-inputs "$LUT_INPUTS" \
  --lut-extra-bit "$LUT_EXTRA_BIT" \
  --magnitude-threshold-init "$MAGNITUDE_THRESHOLD_INIT" \
  --magnitude-threshold-lr "$MAGNITUDE_THRESHOLD_LR" \
  --magnitude-threshold-schedule "$MAGNITUDE_THRESHOLD_SCHEDULE" \
  --magnitude-bit-beta "$MAGNITUDE_BIT_BETA" \
  --magnitude-shadow-epsilon "$MAGNITUDE_SHADOW_EPSILON" \
  --lut-logit-init "$LUT_LOGIT_INIT" \
  --lut-init-mode "$LUT_INIT_MODE" \
  --lut-sign-flip-prob "$LUT_SIGN_FLIP_PROB" \
  --lut-lr "$LUT_LR" \
  --lut-schedule "$LUT_SCHEDULE" \
  --lut-tau-min "$LUT_TAU_MIN" \
  --lut-tau-max "$LUT_TAU_MAX" \
  --lut-hard-epoch-fraction "$LUT_HARD_EPOCH_FRACTION" \
  --lut-hard-transition-epochs "$LUT_HARD_TRANSITION_EPOCHS" \
  --lut-soft-only-epochs "$LUT_SOFT_ONLY_EPOCHS" \
  --lut-hard-bn-only-epochs "$LUT_HARD_BN_ONLY_EPOCHS" \
  "${ANNEAL_ARGS[@]}" \
  "${LUT_SEARCH_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  "$@"
