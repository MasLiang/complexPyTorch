#!/usr/bin/env bash
# FP-only Pol-InSAR representation sweep with distributed spatial validation.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-5}"
WORK_ROOT="${WORK_ROOT:-runs/pol_insar_island/fp_representation_sweep/seed0}"
SEED="${SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FP_EPOCHS="${FP_EPOCHS:-100}"
REPRESENTATIONS="${REPRESENTATIONS:-raw_rms trace log_coherence}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${ROOT_DIR}"
for representation in ${REPRESENTATIONS}; do
  "${PYTHON_BIN}" -m experiments.run_flow \
    --dataset pol_insar_island_t3 --stages fp \
    --data-root "${DATA_ROOT:-datasets_external/pol_insar_island}" \
    --workdir "${WORK_ROOT}/${representation}" \
    --seed "${SEED}" --split-seed "${SPLIT_SEED}" \
    --representation "${representation}" --spatial-block-size "${BLOCK_SIZE}" \
    --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --fp-epochs "${FP_EPOCHS}"
done
