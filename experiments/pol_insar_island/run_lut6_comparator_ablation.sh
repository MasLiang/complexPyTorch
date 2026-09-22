#!/usr/bin/env bash
# Matched LUT6 comparator-gradient ablation using an existing LUT4 parent.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
MODE="${MODE:-stop}"
PARENT_WORKDIR="${PARENT_WORKDIR:-runs/pol_insar_island/log_coherence_block16/seed0}"
WORKDIR="${WORKDIR:-${PARENT_WORKDIR}/lut6_comparator_ablation/${MODE}}"

if [[ "${MODE}" != "stop" && "${MODE}" != "ste" ]]; then
  echo "MODE must be stop or ste, got '${MODE}'." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m experiments.run_flow \
  --dataset pol_insar_island_t3 \
  --stages lut6_residual \
  --skip-parent-stages \
  --parent-workdir "${PARENT_WORKDIR}" \
  --select-any-residual-alpha \
  --workdir "${WORKDIR}" \
  --seed "${SEED:-0}" \
  --split-seed "${SPLIT_SEED:-0}" \
  --data-root "${DATA_ROOT:-datasets_external/pol_insar_island}" \
  --representation "${REPRESENTATION:-log_coherence}" \
  --patch-size "${PATCH_SIZE:-11}" \
  --spatial-block-size "${SPATIAL_BLOCK_SIZE:-16}" \
  --val-fraction "${VAL_FRACTION:-0.15}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --start-filters "${START_FILTERS:-16}" \
  --num-blocks "${NUM_BLOCKS:-3}" \
  --lut6-epochs "${LUT6_EPOCHS:-200}" \
  --dominance-grad-mode "${MODE}" \
  --dominance-ste-margin "${DOMINANCE_STE_MARGIN:-1.0}"
