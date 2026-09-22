#!/usr/bin/env bash
# OpenSARShip SLC: FP -> BiReal-topology FP -> BiReal -> LUT4 -> LUT6.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-5}"
WORKDIR="${WORKDIR:-runs/opensarship_slc/full_flow/seed0}"
SEED="${SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m experiments.run_flow \
  --dataset opensarship_slc \
  --data-root "${DATA_ROOT:-datasets_external/opensarship}" \
  --workdir "${WORKDIR}" --seed "${SEED}" --split-seed "${SPLIT_SEED}" \
  --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --fp-epochs "${FP_EPOCHS:-100}" --shared-fp-epochs "${BIREAL_FP_EPOCHS:-200}" \
  --bireal-epochs "${BIREAL_EPOCHS:-200}" --lut4-epochs "${LUT4_EPOCHS:-200}" \
  --lut6-epochs "${LUT6_EPOCHS:-200}" "$@"
