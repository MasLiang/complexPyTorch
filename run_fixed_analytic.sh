#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${GPU_ID:-0}"
DATADIR="${DATADIR:-${ROOT_DIR}/data}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/bi_workdir/chkpts/Bestmodel_phase1.pt}"
WORKDIR="${WORKDIR:-${ROOT_DIR}/runs/fixed_analytic_dominance_from_phase1_cmpbeta1_e200}"

NUM_EPOCHS="${NUM_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
START_FILTER="${START_FILTER:-11}"
NUM_BLOCKS="${NUM_BLOCKS:-3}"
SPECTRAL_POOL_SCHEME="${SPECTRAL_POOL_SCHEME:-none}"

LR="${LR:-0.01}"
SCHEDULE="${SCHEDULE:-cosine}"
MIN_LR_FACTOR="${MIN_LR_FACTOR:-0.1}"
DOMINANCE_BETA="${DOMINANCE_BETA:-2.0}"
DOMINANCE_PHASE_NORMALIZED="${DOMINANCE_PHASE_NORMALIZED:-0}"
COMPARATOR_BETA="${COMPARATOR_BETA:-1.0}"
COMPARATOR_SCALE="${COMPARATOR_SCALE:-1.0}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.5}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
BN_CALIBRATION_BATCHES="${BN_CALIBRATION_BATCHES:--1}"
OCCUPANCY_INTERVAL="${OCCUPANCY_INTERVAL:-10}"
NO_VALIDATION="${NO_VALIDATION:-0}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "lut_net" ]]; then
    PYTHON_CMD=(python)
else
    PYTHON_CMD=(conda run --no-capture-output -n lut_net python)
fi

mkdir -p "${WORKDIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

echo "==> Fixed analytic dominance experiment on physical GPU ${GPU_ID}"
echo "    Phase1 checkpoint=${CHECKPOINT}"
echo "    workdir=${WORKDIR}"
echo "    training graph: analytic complex multiply + hard local comparator"
echo "    LUT lookup/parameters/backend: disabled; LUT5 is exported only in checkpoints"
echo "    dominance beta=${DOMINANCE_BETA}, comparator beta=${COMPARATOR_BETA}, scale=${COMPARATOR_SCALE}"
echo "    base lr=${LR} (${SCHEDULE}), teacher distillation=${DISTILL_WEIGHT}, BN calibration=${BN_CALIBRATION_BATCHES}"

TRAIN_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --datadir "${DATADIR}"
    --workdir "${WORKDIR}"
    --num-epochs "${NUM_EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --start-filter "${START_FILTER}"
    --num-blocks "${NUM_BLOCKS}"
    --spectral-pool-scheme "${SPECTRAL_POOL_SCHEME}"
    --optimizer sgd
    --lr "${LR}"
    --schedule "${SCHEDULE}"
    --min-lr-factor "${MIN_LR_FACTOR}"
    --dominance-beta "${DOMINANCE_BETA}"
    --comparator-beta "${COMPARATOR_BETA}"
    --comparator-scale "${COMPARATOR_SCALE}"
    --distill-weight "${DISTILL_WEIGHT}"
    --distill-temperature "${DISTILL_TEMPERATURE}"
    --bn-calibration-batches "${BN_CALIBRATION_BATCHES}"
    --occupancy-interval "${OCCUPANCY_INTERVAL}"
    --clipnorm 1.0
    --clipval 1.0
)
if [[ "${DOMINANCE_PHASE_NORMALIZED}" == "1" ]]; then
    TRAIN_ARGS+=(--dominance-phase-normalized)
fi
if [[ "${NO_VALIDATION}" == "1" ]]; then
    TRAIN_ARGS+=(--no-validation)
fi

"${PYTHON_CMD[@]}" "${ROOT_DIR}/training_fixed_analytic.py" "${TRAIN_ARGS[@]}"
