#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_ID=${GPU_ID:-0}
FLOW_ROOT=${FLOW_ROOT:-"$ROOT_DIR/runs/san_francisco/option_a_shortcut_flow"}

GPU_ID="$GPU_ID" \
SHORTCUT_MODE=option_a \
FLOW_ROOT="$FLOW_ROOT" \
  exec "$ROOT_DIR/experiments/san_francisco/run_bireal_flow.sh" "$@"
