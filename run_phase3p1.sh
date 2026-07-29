#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATADIR=${DATADIR:-"$ROOT_DIR/data"}
CONDA_ENV=${CONDA_ENV:-lut_net}
GPU_ID=${GPU_ID:-0}
PHASE3P1_MODE=${PHASE3P1_MODE:-fixed}
if [[ -z "${CHECKPOINT:-}" ]]; then
  if [[ "$PHASE3P1_MODE" == "semantic_lut5" ]]; then
    CHECKPOINT="$ROOT_DIR/runs/phase2p1_c8_octants_semanticste_beta2_bireal_lr01/chkpts/Bestmodel_phase2p1.pt"
  else
    CHECKPOINT="$ROOT_DIR/runs/phase2p1_c8/chkpts/Bestmodel_phase2p1.pt"
  fi
fi

if [[ -z "${WORKDIR:-}" ]]; then
  if [[ "$PHASE3P1_MODE" == "pure_mlp" ]]; then
    WORKDIR="$ROOT_DIR/runs/phase3p1_pure_mlp"
  elif [[ "$PHASE3P1_MODE" == "neural_lut5" ]]; then
    WORKDIR="$ROOT_DIR/runs/phase3p1_neural_lut5"
  elif [[ "$PHASE3P1_MODE" == "semantic_lut5" ]]; then
    WORKDIR="$ROOT_DIR/runs/phase3p1_semantic_lut5"
  elif [[ "$PHASE3P1_MODE" == "learned_lut5" ]]; then
    WORKDIR="$ROOT_DIR/runs/phase3p1_learned_lut5"
  else
    WORKDIR="$ROOT_DIR/runs/phase3p1_c8_local"
  fi
fi

NUM_EPOCHS=${NUM_EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-128}
START_FILTER=${START_FILTER:-11}
NUM_BLOCKS=${NUM_BLOCKS:-3}
OPTIMIZER=${OPTIMIZER:-sgd}
LR=${LR:-0.001}
if [[ -z "${SCHEDULE:-}" ]]; then
  if [[ "$PHASE3P1_MODE" == "learned_lut5" || "$PHASE3P1_MODE" == "semantic_lut5" || "$PHASE3P1_MODE" == "neural_lut5" || "$PHASE3P1_MODE" == "pure_mlp" ]]; then
    SCHEDULE=cosine
  else
    SCHEDULE=default
  fi
fi
MIN_LR_FACTOR=${MIN_LR_FACTOR:-0.1}
C8_BETA=${C8_BETA:-2.0}
if [[ "$PHASE3P1_MODE" == "semantic_lut5" ]]; then
  C8_CODEBOOK=${C8_CODEBOOK:-octants}
  C8_GRAD_MODE=${C8_GRAD_MODE:-semantic_ste}
else
  C8_CODEBOOK=${C8_CODEBOOK:-roots}
  C8_GRAD_MODE=${C8_GRAD_MODE:-softmax}
fi
PHASE3P1_PADDING_MODE=${PHASE3P1_PADDING_MODE:-low_code}
PHASE3P1_TRANSITION_EPOCHS=${PHASE3P1_TRANSITION_EPOCHS:-0}
PHASE3P1_TRANSITION_SCHEDULE=${PHASE3P1_TRANSITION_SCHEDULE:-cosine}
NO_VALIDATION=${NO_VALIDATION:-0}

LUT_SETS=${LUT_SETS:-1}
LUT_ALLOCATION=${LUT_ALLOCATION:-layer}
LUT_SETS_PER_CHANNEL=${LUT_SETS_PER_CHANNEL:-1}
LUT_LOGIT_INIT=${LUT_LOGIT_INIT:-2.0}
LUT_INIT_MODE=${LUT_INIT_MODE:-raw}
LUT_SIGN_FLIP_PROB=${LUT_SIGN_FLIP_PROB:-0}
LUT_LR=${LUT_LR:-0.005}
LUT_SCHEDULE=${LUT_SCHEDULE:-constant}
LUT_TAU_MIN=${LUT_TAU_MIN:-1.0}
LUT_TAU_MAX=${LUT_TAU_MAX:-10.0}
LUT_SOFT_WARMUP_EPOCHS=${LUT_SOFT_WARMUP_EPOCHS:-10}
LUT_ANNEAL_EPOCHS=${LUT_ANNEAL_EPOCHS:-140}
LUT_HARD_TRANSITION_EPOCHS=${LUT_HARD_TRANSITION_EPOCHS:-20}
if [[ -z "${NEURAL_LUT_HIDDEN:-}" ]]; then
  if [[ "$PHASE3P1_MODE" == "pure_mlp" ]]; then
    NEURAL_LUT_HIDDEN=32
  else
    NEURAL_LUT_HIDDEN=8
  fi
fi
NEURAL_LUT_RESIDUAL_SCALE=${NEURAL_LUT_RESIDUAL_SCALE:-1.0}
NEURAL_LUT_HARD_EVAL_INTERVAL=${NEURAL_LUT_HARD_EVAL_INTERVAL:-10}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing Phase 2.1 checkpoint: $CHECKPOINT" >&2
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
if [[ "$PHASE3P1_MODE" == "learned_lut5" || "$PHASE3P1_MODE" == "semantic_lut5" || "$PHASE3P1_MODE" == "neural_lut5" || "$PHASE3P1_MODE" == "pure_mlp" ]]; then
  LUT_ARGS=(
    --lut-sets "$LUT_SETS"
    --lut-allocation "$LUT_ALLOCATION"
    --lut-sets-per-channel "$LUT_SETS_PER_CHANNEL"
    --lut-logit-init "$LUT_LOGIT_INIT"
    --lut-init-mode "$LUT_INIT_MODE"
    --lut-sign-flip-prob "$LUT_SIGN_FLIP_PROB"
    --lut-lr "$LUT_LR"
    --lut-schedule "$LUT_SCHEDULE"
  )
  if [[ "$PHASE3P1_MODE" == "pure_mlp" ]]; then
    LUT_ARGS+=(
      --neural-lut-hidden "$NEURAL_LUT_HIDDEN"
      --neural-lut-hard-eval-interval "$NEURAL_LUT_HARD_EVAL_INTERVAL"
    )
  else
    LUT_ARGS+=(
      --lut-tau-min "$LUT_TAU_MIN"
      --lut-tau-max "$LUT_TAU_MAX"
      --lut-soft-warmup-epochs "$LUT_SOFT_WARMUP_EPOCHS"
      --lut-anneal-epochs "$LUT_ANNEAL_EPOCHS"
      --lut-hard-transition-epochs "$LUT_HARD_TRANSITION_EPOCHS"
    )
  fi
  if [[ "$PHASE3P1_MODE" == "neural_lut5" ]]; then
    LUT_ARGS+=(
      --neural-lut-hidden "$NEURAL_LUT_HIDDEN"
      --neural-lut-residual-scale "$NEURAL_LUT_RESIDUAL_SCALE"
      --neural-lut-hard-eval-interval "$NEURAL_LUT_HARD_EVAL_INTERVAL"
    )
  fi
fi

echo "Running Phase 3.1 mode: $PHASE3P1_MODE"
echo "Checkpoint: $CHECKPOINT"
echo "Workdir: $WORKDIR"
if [[ "$PHASE3P1_MODE" == "pure_mlp" ]]; then
  echo "Operator: pure continuous 5->$NEURAL_LUT_HIDDEN->2 MLP; no trainable LUT entries"
  echo "Initialization: deterministic teacher fit to the Phase2.1 C8 local product"
  echo "Deployment: enumerate and threshold 32 addresses into one hard dual-output LUT5"
  echo "MLP LR/schedule: $LUT_LR/$LUT_SCHEDULE"
  echo "Hard LUT evaluation interval: $NEURAL_LUT_HARD_EVAL_INTERVAL"
  echo "LUT tau/annealing: disabled"
  echo "Padding mode: physical low-code"
elif [[ "$PHASE3P1_MODE" == "neural_lut5" ]]; then
  echo "Operator: residual 5->$NEURAL_LUT_HIDDEN->2 MLP-generated C8 LUT5"
  echo "Deployment: enumerate 32 addresses into one hard dual-output LUT5"
  echo "Neural residual scale: $NEURAL_LUT_RESIDUAL_SCALE"
  echo "Hard projection evaluation interval: $NEURAL_LUT_HARD_EVAL_INTERVAL"
  echo "Padding mode: physical low-code"
  echo "LUT LR/schedule: $LUT_LR/$LUT_SCHEDULE"
  echo "LUT tau: $LUT_TAU_MIN -> $LUT_TAU_MAX; warmup/anneal/hard transition: $LUT_SOFT_WARMUP_EPOCHS/$LUT_ANNEAL_EPOCHS/$LUT_HARD_TRANSITION_EPOCHS"
elif [[ "$PHASE3P1_MODE" == "semantic_lut5" ]]; then
  echo "Operator: direct semantic [sign_r, sign_i, dominance, weight_r, weight_i] -> real/imag LUT5"
  echo "Initialization: exact continuous C8 product with distinct dominance slices"
  echo "Backward: Bi-Real sign STE + sigmoid dominance STE into BN_pre/backbone"
  echo "Padding mode: physical low-code"
  echo "LUT LR/schedule: $LUT_LR/$LUT_SCHEDULE"
  echo "LUT tau: $LUT_TAU_MIN -> $LUT_TAU_MAX; warmup/anneal/hard transition: $LUT_SOFT_WARMUP_EPOCHS/$LUT_ANNEAL_EPOCHS/$LUT_HARD_TRANSITION_EPOCHS"
elif [[ "$PHASE3P1_MODE" == "learned_lut5" ]]; then
  echo "Operator: trainable C8 5-to-2 LUT + local accumulation"
  echo "Padding mode: physical low-code"
  echo "LUT LR/schedule: $LUT_LR/$LUT_SCHEDULE"
  echo "LUT tau: $LUT_TAU_MIN -> $LUT_TAU_MAX; warmup/anneal/hard transition: $LUT_SOFT_WARMUP_EPOCHS/$LUT_ANNEAL_EPOCHS/$LUT_HARD_TRANSITION_EPOCHS"
else
  echo "Operator: decode C8 + binary complex multiply + local real/imag >=0 + accumulate"
  echo "Padding mode: $PHASE3P1_PADDING_MODE"
  echo "Local transition: $PHASE3P1_TRANSITION_EPOCHS epoch(s), $PHASE3P1_TRANSITION_SCHEDULE"
fi
echo "Future mapping: 3-bit activation + 2-bit complex weight -> real/imag LUT5"
echo "C8 codebook: $C8_CODEBOOK"
echo "C8 gradient mode: $C8_GRAD_MODE"
echo "C8 backward beta: $C8_BETA"
echo "LR/schedule: $LR/$SCHEDULE"
echo "Compile: ${COMPILE:-0}"

"${PYTHON_CMD[@]}" "$ROOT_DIR/training.py" \
  --phase 3.1 \
  --phase3p1-mode "$PHASE3P1_MODE" \
  --lut-inputs 5 \
  --c8-beta "$C8_BETA" \
  --c8-codebook "$C8_CODEBOOK" \
  --c8-grad-mode "$C8_GRAD_MODE" \
  --phase3p1-padding-mode "$PHASE3P1_PADDING_MODE" \
  --phase3p1-transition-epochs "$PHASE3P1_TRANSITION_EPOCHS" \
  --phase3p1-transition-schedule "$PHASE3P1_TRANSITION_SCHEDULE" \
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
