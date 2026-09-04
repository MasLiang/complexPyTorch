#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-1}
SPLIT_SEED=${SPLIT_SEED:-0}
RUN_ROOT=${RUN_ROOT:-"$ROOT_DIR/runs/san_francisco/multiseed/seed_${SEED}"}
PYTHON_BIN=${PYTHON_BIN:-python}

GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" SEED="$SEED" SPLIT_SEED="$SPLIT_SEED" \
WORKDIR="$RUN_ROOT/fp_baseline" EPOCHS="${BASELINE_EPOCHS:-100}" \
  "$ROOT_DIR/experiments/san_francisco/run_baseline.sh"

GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" SEED="$SEED" SPLIT_SEED="$SPLIT_SEED" \
WORKDIR="$RUN_ROOT/bireal_flow/bireal_fp" EPOCHS="${FP_EPOCHS:-200}" \
  "$ROOT_DIR/experiments/san_francisco/run_bireal_fp.sh"

GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" SEED="$SEED" SPLIT_SEED="$SPLIT_SEED" \
CHECKPOINT="$RUN_ROOT/bireal_flow/bireal_fp/best.pt" \
WORKDIR="$RUN_ROOT/bireal_flow/bireal" EPOCHS="${BIREAL_EPOCHS:-200}" \
  "$ROOT_DIR/experiments/san_francisco/run_bireal.sh"

GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" SEED="$SEED" SPLIT_SEED="$SPLIT_SEED" \
CHECKPOINT="$RUN_ROOT/bireal_flow/bireal/best.pt" \
WORKDIR="$RUN_ROOT/bireal_flow/bireal_lut" EPOCHS="${LUT4_EPOCHS:-200}" \
  "$ROOT_DIR/experiments/san_francisco/run_bireal_lut.sh"

GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" SEED="$SEED" SPLIT_SEED="$SPLIT_SEED" \
CHECKPOINT="$RUN_ROOT/bireal_flow/bireal_lut/best.pt" \
WORKDIR="$RUN_ROOT/bireal_flow/lut6_residual" EPOCHS="${LUT6_EPOCHS:-200}" \
  "$ROOT_DIR/experiments/san_francisco/run_lut6_residual.sh"
