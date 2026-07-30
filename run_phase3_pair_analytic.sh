#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export TRAIN_FROM_SCRATCH=${TRAIN_FROM_SCRATCH:-1}
export PHASE3_MODE=analytic_pair
export NUM_EPOCHS=${NUM_EPOCHS:-256}
export BATCH_SIZE=${BATCH_SIZE:-256}
export OPTIMIZER=${OPTIMIZER:-adam}
export LR=${LR:-0.01}
export SCHEDULE=${SCHEDULE:-linear}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0}
export CLIPNORM=${CLIPNORM:-0}
export CLIPVAL=${CLIPVAL:-0}
export LUT_KERNEL_MODE=${LUT_KERNEL_MODE:-binary}
export AUGMENTATION=${AUGMENTATION:-real_lut}
export LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
export NO_VALIDATION=${NO_VALIDATION:-1}
export PRE_BN_MODE=${PRE_BN_MODE:-none}
export POST_BN_MODE=${POST_BN_MODE:-covariance}

exec "$ROOT_DIR/run_phase3.sh" "$@"
