#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${GPU_ID:-0}"
ENCODER_MODE="${ENCODER_MODE:-phase}"
DATADIR="${DATADIR:-${ROOT_DIR}/data}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/bi_workdir/chkpts/Bestmodel_phase1.pt}"
WORKDIR="${WORKDIR:-${ROOT_DIR}/runs/direct_lut5_${ENCODER_MODE}_from_phase1_hard_e200}"

NUM_EPOCHS="${NUM_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
START_FILTER="${START_FILTER:-11}"
NUM_BLOCKS="${NUM_BLOCKS:-3}"
SPECTRAL_POOL_SCHEME="${SPECTRAL_POOL_SCHEME:-none}"

LR="${LR:-0.01}"
SCHEDULE="${SCHEDULE:-cosine}"
MIN_LR_FACTOR="${MIN_LR_FACTOR:-0.1}"
LUT_LR="${LUT_LR:-0.002}"
LUT_SCHEDULE="${LUT_SCHEDULE:-follow-base}"
LUT_FREEZE_EPOCHS="${LUT_FREEZE_EPOCHS:-60}"
INITIAL_LOGIT_MARGIN="${INITIAL_LOGIT_MARGIN:-0.25}"
LUT_MARGIN="${LUT_MARGIN:-0.15}"
LUT_MARGIN_WEIGHT="${LUT_MARGIN_WEIGHT:-0.0001}"
C8_BETA="${C8_BETA:-2.0}"

LUT_ALLOCATION="${LUT_ALLOCATION:-layer}"
LUT_SETS="${LUT_SETS:-1}"
LUT_SETS_PER_CHANNEL="${LUT_SETS_PER_CHANNEL:-1}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.5}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
BN_CALIBRATION_BATCHES="${BN_CALIBRATION_BATCHES:-100}"
OCCUPANCY_INTERVAL="${OCCUPANCY_INTERVAL:-10}"
NO_VALIDATION="${NO_VALIDATION:-0}"

if [[ "${ENCODER_MODE}" != "phase" && "${ENCODER_MODE}" != "dominance" ]]; then
    echo "ENCODER_MODE must be phase or dominance" >&2
    exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "lut_net" ]]; then
    PYTHON_CMD=(python)
else
    PYTHON_CMD=(conda run --no-capture-output -n lut_net python)
fi

mkdir -p "${WORKDIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

echo "==> Direct LUT5 ${ENCODER_MODE} experiment on physical GPU ${GPU_ID}"
echo "    Phase1 checkpoint=${CHECKPOINT}"
echo "    workdir=${WORKDIR}"
echo "    hard 5-to-2 forward starts at epoch 0; LUT frozen for ${LUT_FREEZE_EPOCHS} epoch(s)"
echo "    base lr=${LR} (${SCHEDULE}), LUT lr=${LUT_LR} (${LUT_SCHEDULE}), teacher distillation=${DISTILL_WEIGHT}"

TRAIN_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --encoder-mode "${ENCODER_MODE}"
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
    --lut-lr "${LUT_LR}"
    --lut-schedule "${LUT_SCHEDULE}"
    --lut-freeze-epochs "${LUT_FREEZE_EPOCHS}"
    --initial-logit-margin "${INITIAL_LOGIT_MARGIN}"
    --lut-margin "${LUT_MARGIN}"
    --lut-margin-weight "${LUT_MARGIN_WEIGHT}"
    --c8-beta "${C8_BETA}"
    --lut-allocation "${LUT_ALLOCATION}"
    --lut-sets "${LUT_SETS}"
    --lut-sets-per-channel "${LUT_SETS_PER_CHANNEL}"
    --distill-weight "${DISTILL_WEIGHT}"
    --distill-temperature "${DISTILL_TEMPERATURE}"
    --bn-calibration-batches "${BN_CALIBRATION_BATCHES}"
    --occupancy-interval "${OCCUPANCY_INTERVAL}"
    --clipnorm 1.0
    --clipval 1.0
)
if [[ "${NO_VALIDATION}" == "1" ]]; then
    TRAIN_ARGS+=(--no-validation)
fi

"${PYTHON_CMD[@]}" "${ROOT_DIR}/training_direct_lut5.py" "${TRAIN_ARGS[@]}"
