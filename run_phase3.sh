#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export GPU_ID=${GPU_ID:-0}
export DATADIR=${DATADIR:-"$ROOT_DIR/data"}
export WORKDIR=${WORKDIR:-"$ROOT_DIR/runs/phase3_pair_lut4"}

NUM_EPOCHS=${NUM_EPOCHS:-256}
BATCH_SIZE=${BATCH_SIZE:-256}
START_FILTER=${START_FILTER:-16}
NUM_BLOCKS=${NUM_BLOCKS:-3}
LR=${LR:-0.02}
SCHEDULE=${SCHEDULE:-multistep}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
POST_BN_MODE=${POST_BN_MODE:-covariance}
PHASE3_OPERATOR=${PHASE3_OPERATOR:-pair_lut4}
PAIR_LUT_PARAMETERIZATION=${PAIR_LUT_PARAMETERIZATION:-independent}
LUT_INPUTS=${LUT_INPUTS:-4}
PAIR_LUT_ENCODING=${PAIR_LUT_ENCODING:-standard}
DOMINANCE_GRAD_MODE=${DOMINANCE_GRAD_MODE:-stop}
DOMINANCE_RESIDUAL_LR=${DOMINANCE_RESIDUAL_LR:-0.002}
DOMINANCE_RESIDUAL_ALPHA_START=${DOMINANCE_RESIDUAL_ALPHA_START:-0.1}
DOMINANCE_RESIDUAL_ALPHA_END=${DOMINANCE_RESIDUAL_ALPHA_END:-1.0}
DOMINANCE_RESIDUAL_RAMP_EPOCHS=${DOMINANCE_RESIDUAL_RAMP_EPOCHS:-120}
DOMINANCE_BASE_LR=${DOMINANCE_BASE_LR:-0}
DOMINANCE_BASE_ALPHA_START=${DOMINANCE_BASE_ALPHA_START:-0}
DOMINANCE_BASE_ALPHA_END=${DOMINANCE_BASE_ALPHA_END:-0}
DOMINANCE_BASE_RAMP_EPOCHS=${DOMINANCE_BASE_RAMP_EPOCHS:-1}
DOMINANCE_STE_MARGIN=${DOMINANCE_STE_MARGIN:-1.0}
NUM_WORKERS=${NUM_WORKERS:-8}
LOG_INTERVAL=${LOG_INTERVAL:-0}
ARGS=(
  --phase 3
  --phase3-operator "$PHASE3_OPERATOR"
  --num-epochs "$NUM_EPOCHS"
  --pair-lut-parameterization "$PAIR_LUT_PARAMETERIZATION"
  --pair-lut-inputs "$LUT_INPUTS"
  --pair-lut-encoding "$PAIR_LUT_ENCODING"
  --dominance-grad-mode "$DOMINANCE_GRAD_MODE"
  --dominance-residual-lr "$DOMINANCE_RESIDUAL_LR"
  --dominance-residual-alpha-start "$DOMINANCE_RESIDUAL_ALPHA_START"
  --dominance-residual-alpha-end "$DOMINANCE_RESIDUAL_ALPHA_END"
  --dominance-residual-ramp-epochs "$DOMINANCE_RESIDUAL_RAMP_EPOCHS"
  --dominance-base-lr "$DOMINANCE_BASE_LR"
  --dominance-base-alpha-start "$DOMINANCE_BASE_ALPHA_START"
  --dominance-base-alpha-end "$DOMINANCE_BASE_ALPHA_END"
  --dominance-base-ramp-epochs "$DOMINANCE_BASE_RAMP_EPOCHS"
  --dominance-ste-margin "$DOMINANCE_STE_MARGIN"
  --batch-size "$BATCH_SIZE"
  --start-filter "$START_FILTER"
  --num-blocks "$NUM_BLOCKS"
  --optimizer adam
  --lr "$LR"
  --schedule "$SCHEDULE"
  --weight-decay "$WEIGHT_DECAY"
  --clipnorm "${CLIPNORM:-0}"
  --clipval "${CLIPVAL:-0}"
  --spectral-pool-scheme none
  --activation-grad-mode bireal
  --post-bn-mode "$POST_BN_MODE"
  --augmentation real_lut
  --label-smoothing 0.1
  --no-validation
  --num-workers "$NUM_WORKERS"
  --log-interval "$LOG_INTERVAL"
)

if [[ -n "${CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
else
  ARGS+=(--train-from-scratch)
fi

printf 'Running configurable LUT-neuron complex Bi-Real Phase 3 on GPU %s\n' "$GPU_ID"
printf 'LUT activation encoding: exact 0/1 bits\n'
printf 'PairLUT4 parameterization: %s\n' "$PAIR_LUT_PARAMETERIZATION"
if [[ "$PAIR_LUT_ENCODING" == "dominance" ]]; then
  printf 'Complex LUT inputs: 6 bits (2 complex activations with parallel dominance bits)\n'
  printf 'Dominance gradient mode: %s\n' "$DOMINANCE_GRAD_MODE"
  if [[ "$DOMINANCE_GRAD_MODE" == "ste" ]]; then
    printf 'Dominance STE margin: %s\n' "$DOMINANCE_STE_MARGIN"
  fi
  if [[ "$PAIR_LUT_PARAMETERIZATION" == "categorical_residual" ]]; then
    printf "Dominance residual: lr=%s alpha=%s->%s ramp=%s epochs\n" \
      "$DOMINANCE_RESIDUAL_LR" "$DOMINANCE_RESIDUAL_ALPHA_START" \
      "$DOMINANCE_RESIDUAL_ALPHA_END" "$DOMINANCE_RESIDUAL_RAMP_EPOCHS"
    printf "LUT4 common correction: lr=%s alpha=%s->%s ramp=%s epochs\n" \
      "$DOMINANCE_BASE_LR" "$DOMINANCE_BASE_ALPHA_START" \
      "$DOMINANCE_BASE_ALPHA_END" "$DOMINANCE_BASE_RAMP_EPOCHS"
  fi
else
  printf 'Complex LUT inputs: %s bits (%s complex activations per group)\n' "$LUT_INPUTS" "$((LUT_INPUTS / 2))"
fi
printf 'Pair LUT activation encoding: %s\n' "$PAIR_LUT_ENCODING"
if [[ -n "${CHECKPOINT:-}" && "$PAIR_LUT_ENCODING" == "dominance" ]]; then
  printf 'LUT initialization: expand Phase3 LUT4 checkpoint, or keep random LUTs for Phase2 checkpoint\n'
elif [[ -n "${CHECKPOINT:-}" ]]; then
  printf 'LUT initialization: compile Phase 2 binary complex weights\n'
elif [[ "$PAIR_LUT_PARAMETERIZATION" == categorical* ]]; then
  printf 'LUT initialization: small random categorical logits\n'
else
  printf 'LUT initialization: bimodal independent logits\n'
fi
if [[ "$PAIR_LUT_PARAMETERIZATION" == categorical* ]]; then
  printf 'LUT table proxy: hard argmax forward, softmax STE backward, tau=1.0\n'
else
  printf 'LUT table proxy: independent hard threshold forward, identity STE backward\n'
fi
printf 'LUT training flow: hardware-hard table from epoch 1\n'
printf 'Workdir: %s\n' "$WORKDIR"
exec "$ROOT_DIR/run_training.sh" "${ARGS[@]}" "$@"
