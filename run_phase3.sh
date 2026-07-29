#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export GPU_ID=${GPU_ID:-0}
export DATADIR=${DATADIR:-"$ROOT_DIR/data"}
export WORKDIR=${WORKDIR:-"$ROOT_DIR/runs/phase3_pair_lut"}

NUM_EPOCHS=${NUM_EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-128}
START_FILTER=${START_FILTER:-11}
NUM_BLOCKS=${NUM_BLOCKS:-3}
if [[ "${TRAIN_FROM_SCRATCH:-0}" == "1" ]]; then
  DEFAULT_LR=0.01
else
  DEFAULT_LR=0.001
fi
LR=${LR:-$DEFAULT_LR}
SCHEDULE=${SCHEDULE:-cosine}
MIN_LR_FACTOR=${MIN_LR_FACTOR:-0.01}
LUT_LR=${LUT_LR:-0.01}
LUT_SCHEDULE=${LUT_SCHEDULE:-constant}
LUT_TRAINING_MODE=${LUT_TRAINING_MODE:-anneal}
LUT_KERNEL_MODE=${LUT_KERNEL_MODE:-auto}
LUT_INIT_MODE=${LUT_INIT_MODE:-normal}
LUT_LOGIT_INIT=${LUT_LOGIT_INIT:-1.0}
LUT_BIMODAL_NEGATIVE_MEAN=${LUT_BIMODAL_NEGATIVE_MEAN:--1.0}
LUT_BIMODAL_NEGATIVE_STD=${LUT_BIMODAL_NEGATIVE_STD:-0.2}
LUT_BIMODAL_POSITIVE_MEAN=${LUT_BIMODAL_POSITIVE_MEAN:-1.0}
LUT_BIMODAL_POSITIVE_STD=${LUT_BIMODAL_POSITIVE_STD:-0.1}
LUT_TAU_MIN=${LUT_TAU_MIN:-0.5}
LUT_TAU_MAX=${LUT_TAU_MAX:-10.0}
LUT_ANNEAL_EPOCHS=${LUT_ANNEAL_EPOCHS:-160}
LUT_HARD_TRANSITION_EPOCHS=${LUT_HARD_TRANSITION_EPOCHS:-40}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
OPTIMIZER=${OPTIMIZER:-sgd}
CLIPNORM=${CLIPNORM:-1.0}
CLIPVAL=${CLIPVAL:-1.0}
AUGMENTATION=${AUGMENTATION:-complex_default}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}
SPECTRAL_POOL_SCHEME=${SPECTRAL_POOL_SCHEME:-none}
BINARY_WEIGHT_SCALE=${BINARY_WEIGHT_SCALE:-channel}
PRE_BN_MODE=${PRE_BN_MODE:-covariance}
POST_BN_MODE=${POST_BN_MODE:-covariance}

ARGS=(
  --phase 3
  --num-epochs "$NUM_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --start-filter "$START_FILTER"
  --num-blocks "$NUM_BLOCKS"
  --lr "$LR"
  --optimizer "$OPTIMIZER"
  --clipnorm "$CLIPNORM"
  --clipval "$CLIPVAL"
  --schedule "$SCHEDULE"
  --min-lr-factor "$MIN_LR_FACTOR"
  --lut-lr "$LUT_LR"
  --lut-schedule "$LUT_SCHEDULE"
  --lut-training-mode "$LUT_TRAINING_MODE"
  --lut-kernel-mode "$LUT_KERNEL_MODE"
  --lut-init-mode "$LUT_INIT_MODE"
  --lut-logit-init "$LUT_LOGIT_INIT"
  --lut-bimodal-negative-mean "$LUT_BIMODAL_NEGATIVE_MEAN"
  --lut-bimodal-negative-std "$LUT_BIMODAL_NEGATIVE_STD"
  --lut-bimodal-positive-mean "$LUT_BIMODAL_POSITIVE_MEAN"
  --lut-bimodal-positive-std "$LUT_BIMODAL_POSITIVE_STD"
  --lut-tau-min "$LUT_TAU_MIN"
  --lut-tau-max "$LUT_TAU_MAX"
  --lut-anneal-epochs "$LUT_ANNEAL_EPOCHS"
  --lut-hard-transition-epochs "$LUT_HARD_TRANSITION_EPOCHS"
  --weight-decay "$WEIGHT_DECAY"
  --spectral-pool-scheme "$SPECTRAL_POOL_SCHEME"
  --binary-weight-scale "$BINARY_WEIGHT_SCALE"
  --pre-bn-mode "$PRE_BN_MODE"
  --post-bn-mode "$POST_BN_MODE"
  --augmentation "$AUGMENTATION"
  --label-smoothing "$LABEL_SMOOTHING"
)

if [[ "${NO_VALIDATION:-0}" == "1" ]]; then
  ARGS+=(--no-validation)
fi

if [[ "${TRAIN_FROM_SCRATCH:-0}" == "1" ]]; then
  ARGS+=(--train-from-scratch)
else
  CHECKPOINT=${CHECKPOINT:-"$ROOT_DIR/bi_workdir/chkpts/Bestmodel_phase2.pt"}
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

printf 'Running Phase 3 pair-LUT neurons on GPU %s\n' "$GPU_ID"
if [[ "${TRAIN_FROM_SCRATCH:-0}" == "1" ]]; then
  if [[ "$LUT_INIT_MODE" == "bimodal" ]]; then
    printf 'Initialization: bimodal N(%s,%s) / N(%s,%s)\n' \
      "$LUT_BIMODAL_NEGATIVE_MEAN" "$LUT_BIMODAL_NEGATIVE_STD" \
      "$LUT_BIMODAL_POSITIVE_MEAN" "$LUT_BIMODAL_POSITIVE_STD"
  else
    printf 'Initialization: random normal (std=%s)\n' "$LUT_LOGIT_INIT"
  fi
else
  printf 'Checkpoint: %s\n' "$CHECKPOINT"
fi
printf 'Workdir: %s\n' "$WORKDIR"
printf 'LUT training mode: %s\n' "$LUT_TRAINING_MODE"
printf 'LUT kernel mode: %s\n' "$LUT_KERNEL_MODE"
printf 'Residual BN modes: pre=%s, post=%s\n' \
  "$PRE_BN_MODE" "$POST_BN_MODE"
if [[ "$LUT_TRAINING_MODE" == "anneal" ]]; then
  printf 'Annealing: %s soft epochs + %s hard-transition epochs\n' \
    "$LUT_ANNEAL_EPOCHS" "$LUT_HARD_TRANSITION_EPOCHS"
fi
exec "$ROOT_DIR/run_training.sh" "${ARGS[@]}" "$@"
