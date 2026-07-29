#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${GPU_ID:-0}"
DATADIR="${DATADIR:-${ROOT_DIR}/data}"
WORKDIR="${WORKDIR:-${ROOT_DIR}/runs/mlp_flow_layer1}"
PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT:-${ROOT_DIR}/bi_workdir/chkpts/Bestmodel_phase1.pt}"
START_FLOW_PHASE="${START_FLOW_PHASE:-2.2}"
END_FLOW_PHASE="${END_FLOW_PHASE:-4.2}"
TRANSITION_CHECKPOINT_KIND="${TRANSITION_CHECKPOINT_KIND:-best}"
PHASE2P2_CHECKPOINT="${PHASE2P2_CHECKPOINT:-}"
PHASE3P2_CHECKPOINT="${PHASE3P2_CHECKPOINT:-}"

OPERATION_ALLOCATION="${OPERATION_ALLOCATION:-layer}"
OPERATION_SETS="${OPERATION_SETS:-1}"
OPERATION_SETS_PER_CHANNEL="${OPERATION_SETS_PER_CHANNEL:-1}"
DOMINANCE_BETA="${DOMINANCE_BETA:-2.0}"
BINARY_MLP_HIDDEN="${BINARY_MLP_HIDDEN:-64}"
BINARY_MLP_LOGIT_INIT="${BINARY_MLP_LOGIT_INIT:-0.5}"
LUT_LOGIT_INIT="${LUT_LOGIT_INIT:-2.0}"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
START_FILTER="${START_FILTER:-11}"
NUM_BLOCKS="${NUM_BLOCKS:-3}"
SPECTRAL_POOL_SCHEME="${SPECTRAL_POOL_SCHEME:-none}"

PHASE2P2_EPOCHS="${PHASE2P2_EPOCHS:-200}"
PHASE3P2_EPOCHS="${PHASE3P2_EPOCHS:-200}"
PHASE4P2_EPOCHS="${PHASE4P2_EPOCHS:-200}"

PHASE2P2_LR="${PHASE2P2_LR:-0.001}"
PHASE2P2_OPERATION_LR="${PHASE2P2_OPERATION_LR:-0.001}"
PHASE2P2_SCHEDULE="${PHASE2P2_SCHEDULE:-default}"
PHASE2P2_OPERATION_SCHEDULE="${PHASE2P2_OPERATION_SCHEDULE:-constant}"
PHASE3P2_LR="${PHASE3P2_LR:-0.001}"
PHASE3P2_OPERATION_LR="${PHASE3P2_OPERATION_LR:-0.001}"
PHASE3P2_SCHEDULE="${PHASE3P2_SCHEDULE:-default}"
PHASE3P2_OPERATION_SCHEDULE="${PHASE3P2_OPERATION_SCHEDULE:-constant}"
PHASE3P2_SCORE_SURROGATE="${PHASE3P2_SCORE_SURROGATE:-0}"
PHASE3P2_FREEZE_OPERATION_EPOCHS="${PHASE3P2_FREEZE_OPERATION_EPOCHS:-0}"
PHASE3P2_BN_CALIBRATION_BATCHES="${PHASE3P2_BN_CALIBRATION_BATCHES:-0}"
PHASE4P2_LR="${PHASE4P2_LR:-0.001}"
PHASE4P2_LUT_LR="${PHASE4P2_LUT_LR:-0.001}"
PHASE4P2_SCHEDULE="${PHASE4P2_SCHEDULE:-cosine}"
PHASE4P2_OPERATION_SCHEDULE="${PHASE4P2_OPERATION_SCHEDULE:-constant}"

LUT_TAU_MIN="${LUT_TAU_MIN:-1.0}"
LUT_TAU_MAX="${LUT_TAU_MAX:-10.0}"
LUT_SOFT_WARMUP_EPOCHS="${LUT_SOFT_WARMUP_EPOCHS:-10}"
LUT_ANNEAL_EPOCHS="${LUT_ANNEAL_EPOCHS:-140}"
LUT_HARD_TRANSITION_EPOCHS="${LUT_HARD_TRANSITION_EPOCHS:-20}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "lut_net" ]]; then
    PYTHON_CMD=(python)
else
    PYTHON_CMD=(conda run --no-capture-output -n lut_net python)
fi

case "${TRANSITION_CHECKPOINT_KIND}" in
    last) TRANSITION_PREFIX="Lastmodel" ;;
    best) TRANSITION_PREFIX="Bestmodel" ;;
    *)
        echo "TRANSITION_CHECKPOINT_KIND must be 'last' or 'best'" >&2
        exit 2
        ;;
esac

mkdir -p "${WORKDIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
    --datadir "${DATADIR}"
    --workdir "${WORKDIR}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --start-filter "${START_FILTER}"
    --num-blocks "${NUM_BLOCKS}"
    --spectral-pool-scheme "${SPECTRAL_POOL_SCHEME}"
    --operation-allocation "${OPERATION_ALLOCATION}"
    --operation-sets "${OPERATION_SETS}"
    --operation-sets-per-channel "${OPERATION_SETS_PER_CHANNEL}"
    --dominance-beta "${DOMINANCE_BETA}"
    --binary-mlp-hidden "${BINARY_MLP_HIDDEN}"
    --binary-mlp-logit-init "${BINARY_MLP_LOGIT_INIT}"
    --lut-logit-init "${LUT_LOGIT_INIT}"
    --optimizer sgd
    --min-lr-factor 0.1
    --clipnorm 1.0
    --clipval 1.0
)

phase_rank() {
    case "$1" in
        2.2) echo 22 ;;
        3.2) echo 32 ;;
        4.2) echo 42 ;;
        *) echo "Unknown flow phase: $1" >&2; exit 2 ;;
    esac
}

run_phase() {
    local phase="$1"
    local checkpoint="$2"
    local epochs="$3"
    local lr="$4"
    local operation_lr="$5"
    local schedule="$6"
    shift 6

    echo "==> MLP flow Phase ${phase} on physical GPU ${GPU_ID}"
    echo "    allocation=${OPERATION_ALLOCATION}, operation_sets=${OPERATION_SETS}, sets_per_channel=${OPERATION_SETS_PER_CHANNEL}"
    echo "    checkpoint=${checkpoint}"
    "${PYTHON_CMD[@]}" "${ROOT_DIR}/training_mlp_flow.py" \
        --flow-phase "${phase}" \
        --checkpoint "${checkpoint}" \
        --num-epochs "${epochs}" \
        --lr "${lr}" \
        --operation-lr "${operation_lr}" \
        --schedule "${schedule}" \
        "${COMMON_ARGS[@]}" \
        "$@"
}

START_RANK="$(phase_rank "${START_FLOW_PHASE}")"
END_RANK="$(phase_rank "${END_FLOW_PHASE}")"
if (( START_RANK > END_RANK )); then
    echo "START_FLOW_PHASE must not be later than END_FLOW_PHASE" >&2
    exit 2
fi

if (( START_RANK <= 22 && END_RANK >= 22 )); then
    run_phase 2.2 "${PHASE1_CHECKPOINT}" "${PHASE2P2_EPOCHS}" \
        "${PHASE2P2_LR}" "${PHASE2P2_OPERATION_LR}" "${PHASE2P2_SCHEDULE}" \
        --operation-schedule "${PHASE2P2_OPERATION_SCHEDULE}"
fi

if [[ -z "${PHASE2P2_CHECKPOINT}" ]]; then
    PHASE2P2_CHECKPOINT="${WORKDIR}/chkpts/${TRANSITION_PREFIX}_phase2p2.pt"
fi
PHASE3P2_EXTRA_ARGS=(
    --operation-schedule "${PHASE3P2_OPERATION_SCHEDULE}"
    --phase3p2-freeze-operation-epochs "${PHASE3P2_FREEZE_OPERATION_EPOCHS}"
    --phase3p2-bn-calibration-batches "${PHASE3P2_BN_CALIBRATION_BATCHES}"
)
case "${PHASE3P2_SCORE_SURROGATE}" in
    1|true|TRUE|yes|YES)
        PHASE3P2_EXTRA_ARGS+=(--phase3p2-score-surrogate)
        ;;
    0|false|FALSE|no|NO) ;;
    *)
        echo "PHASE3P2_SCORE_SURROGATE must be a boolean value" >&2
        exit 2
        ;;
esac
if (( START_RANK <= 32 && END_RANK >= 32 )); then
    run_phase 3.2 "${PHASE2P2_CHECKPOINT}" "${PHASE3P2_EPOCHS}" \
        "${PHASE3P2_LR}" "${PHASE3P2_OPERATION_LR}" "${PHASE3P2_SCHEDULE}" \
        "${PHASE3P2_EXTRA_ARGS[@]}"
fi

if [[ -z "${PHASE3P2_CHECKPOINT}" ]]; then
    PHASE3P2_CHECKPOINT="${WORKDIR}/chkpts/${TRANSITION_PREFIX}_phase3p2.pt"
fi
if (( START_RANK <= 42 && END_RANK >= 42 )); then
    run_phase 4.2 "${PHASE3P2_CHECKPOINT}" "${PHASE4P2_EPOCHS}" \
        "${PHASE4P2_LR}" "${PHASE3P2_OPERATION_LR}" "${PHASE4P2_SCHEDULE}" \
        --operation-schedule "${PHASE4P2_OPERATION_SCHEDULE}" \
        --lut-lr "${PHASE4P2_LUT_LR}" \
        --lut-schedule constant \
        --lut-tau-min "${LUT_TAU_MIN}" \
        --lut-tau-max "${LUT_TAU_MAX}" \
        --lut-soft-warmup-epochs "${LUT_SOFT_WARMUP_EPOCHS}" \
        --lut-anneal-epochs "${LUT_ANNEAL_EPOCHS}" \
        --lut-hard-transition-epochs "${LUT_HARD_TRANSITION_EPOCHS}"
fi

"${PYTHON_CMD[@]}" "${ROOT_DIR}/scripts/analyze_mlp_flow.py" "${WORKDIR}"
echo "==> Completed MLP flow: ${WORKDIR}"
