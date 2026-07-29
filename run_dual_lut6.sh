#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${GPU_ID:-0}"
DATADIR="${DATADIR:-${ROOT_DIR}/data}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/bi_workdir/chkpts/Bestmodel_phase2.pt}"
WORKDIR="${WORKDIR:-${ROOT_DIR}/runs/phase2p7_dual_lut6_2bit_rho80_cosine_e200}"

NUM_EPOCHS="${NUM_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
START_FILTER="${START_FILTER:-11}"
NUM_BLOCKS="${NUM_BLOCKS:-3}"
SPECTRAL_POOL_SCHEME="${SPECTRAL_POOL_SCHEME:-none}"

LR="${LR:-0.01}"
THRESHOLD_LR="${THRESHOLD_LR:-0.001}"
SCHEDULE="${SCHEDULE:-cosine}"
MIN_LR_FACTOR="${MIN_LR_FACTOR:-0.1}"
THRESHOLD_INIT="${THRESHOLD_INIT:-0.675}"
MAGNITUDE_BETA="${MAGNITUDE_BETA:-4.0}"
HIGH_RATIO="${HIGH_RATIO:-3.0}"
RHO_WARMUP_EPOCHS="${RHO_WARMUP_EPOCHS:-10}"
RHO_TRANSITION_EPOCHS="${RHO_TRANSITION_EPOCHS:-80}"
RHO_SCHEDULE="${RHO_SCHEDULE:-cosine}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "lut_net" ]]; then
    PYTHON_CMD=(python)
else
    PYTHON_CMD=(conda run --no-capture-output -n lut_net python)
fi

mkdir -p "${WORKDIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

echo "==> Phase2.7 dual LUT6_2 experiment on physical GPU ${GPU_ID}"
echo "    checkpoint=${CHECKPOINT}"
echo "    workdir=${WORKDIR}"
echo "    rho=0 for ${RHO_WARMUP_EPOCHS} epoch(s), then ${RHO_SCHEDULE} to rho=1 over ${RHO_TRANSITION_EPOCHS} epoch(s)"
echo "    activation levels use high:low ratio ${HIGH_RATIO}; there is no local output truncation"

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
    --threshold-lr "${THRESHOLD_LR}"
    --schedule "${SCHEDULE}"
    --min-lr-factor "${MIN_LR_FACTOR}"
    --threshold-init "${THRESHOLD_INIT}"
    --magnitude-beta "${MAGNITUDE_BETA}"
    --high-ratio "${HIGH_RATIO}"
    --rho-warmup-epochs "${RHO_WARMUP_EPOCHS}"
    --rho-transition-epochs "${RHO_TRANSITION_EPOCHS}"
    --rho-schedule "${RHO_SCHEDULE}"
    --clipnorm 1.0
    --clipval 1.0
)
"${PYTHON_CMD[@]}" "${ROOT_DIR}/training_dual_lut6.py" "${TRAIN_ARGS[@]}"

EXPORT_ARGS=(
    "${WORKDIR}/chkpts/Bestmodel_phase2p7.pt"
    --output "${WORKDIR}/dual_lut6_hardware.json"
)
"${PYTHON_CMD[@]}" "${ROOT_DIR}/scripts/export_dual_lut6.py" "${EXPORT_ARGS[@]}"
