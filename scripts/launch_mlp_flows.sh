#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LAYER_GPU="${LAYER_GPU:-3}"
CHANNEL_GPU="${CHANNEL_GPU:-5}"
LAYER_WORKDIR="${LAYER_WORKDIR:-${ROOT_DIR}/runs/mlp_flow_layer1_full}"
CHANNEL_WORKDIR="${CHANNEL_WORKDIR:-${ROOT_DIR}/runs/mlp_flow_channel1_full}"

if [[ "${LAYER_GPU}" == "${CHANNEL_GPU}" ]]; then
    echo "LAYER_GPU and CHANNEL_GPU must be different" >&2
    exit 2
fi

gpu_memory_used() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F, -v target="$1" '$1 + 0 == target {gsub(/ /, "", $2); print $2}'
}

for gpu in "${LAYER_GPU}" "${CHANNEL_GPU}"; do
    used="$(gpu_memory_used "${gpu}")"
    if [[ -z "${used}" ]]; then
        echo "GPU ${gpu} was not found" >&2
        exit 2
    fi
    if (( used > 1024 )); then
        echo "GPU ${gpu} is not idle (${used} MiB already used)" >&2
        exit 3
    fi
done

mkdir -p "${LAYER_WORKDIR}" "${CHANNEL_WORKDIR}"

nohup setsid env \
    GPU_ID="${LAYER_GPU}" \
    WORKDIR="${LAYER_WORKDIR}" \
    OPERATION_ALLOCATION=layer \
    OPERATION_SETS=1 \
    OPERATION_SETS_PER_CHANNEL=1 \
    "${ROOT_DIR}/run_mlp_flow.sh" \
    >"${LAYER_WORKDIR}/launcher.log" 2>&1 &
LAYER_PID=$!
echo "${LAYER_PID}" >"${LAYER_WORKDIR}/launcher.pid"

nohup setsid env \
    GPU_ID="${CHANNEL_GPU}" \
    WORKDIR="${CHANNEL_WORKDIR}" \
    OPERATION_ALLOCATION=channel \
    OPERATION_SETS=1 \
    OPERATION_SETS_PER_CHANNEL=1 \
    "${ROOT_DIR}/run_mlp_flow.sh" \
    >"${CHANNEL_WORKDIR}/launcher.log" 2>&1 &
CHANNEL_PID=$!
echo "${CHANNEL_PID}" >"${CHANNEL_WORKDIR}/launcher.pid"

echo "Layer-wise flow:   GPU ${LAYER_GPU}, PID ${LAYER_PID}, ${LAYER_WORKDIR}"
echo "Channel-wise flow: GPU ${CHANNEL_GPU}, PID ${CHANNEL_PID}, ${CHANNEL_WORKDIR}"
