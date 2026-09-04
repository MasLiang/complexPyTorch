#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${RUN_ROOT:-"$ROOT_DIR/runs/san_francisco/full_flow"}
GPU_ID=${GPU_ID:-0}

GPU_ID="$GPU_ID" \
WORKDIR="$RUN_ROOT/fp_baseline" \
EPOCHS="${BASELINE_EPOCHS:-100}" \
  "$ROOT_DIR/experiments/san_francisco/run_baseline.sh"

GPU_ID="$GPU_ID" \
FLOW_ROOT="$RUN_ROOT/bireal_flow" \
  "$ROOT_DIR/experiments/san_francisco/run_bireal_flow.sh"
