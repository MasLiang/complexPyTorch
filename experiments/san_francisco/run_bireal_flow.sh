#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FLOW_ROOT=${FLOW_ROOT:-"$ROOT_DIR/runs/san_francisco/bireal_flow"}

GPU_ID="${GPU_ID:-0}" WORKDIR="$FLOW_ROOT/bireal_fp" \
  "$ROOT_DIR/experiments/san_francisco/run_bireal_fp.sh" "$@"

GPU_ID="${GPU_ID:-0}" \
CHECKPOINT="$FLOW_ROOT/bireal_fp/best.pt" \
WORKDIR="$FLOW_ROOT/bireal" \
  "$ROOT_DIR/experiments/san_francisco/run_bireal.sh" "$@"

GPU_ID="${GPU_ID:-0}" \
CHECKPOINT="$FLOW_ROOT/bireal/best.pt" \
WORKDIR="$FLOW_ROOT/bireal_lut" \
  "$ROOT_DIR/experiments/san_francisco/run_bireal_lut.sh" "$@"


GPU_ID="${GPU_ID:-0}" \
CHECKPOINT="$FLOW_ROOT/bireal_lut/best.pt" \
WORKDIR="$FLOW_ROOT/lut6_residual" \
  "$ROOT_DIR/experiments/san_francisco/run_lut6_residual.sh" "$@"

GPU_ID="${GPU_ID:-0}" \
CHECKPOINT="$FLOW_ROOT/lut6_residual/best.pt" \
WORKDIR="$FLOW_ROOT/lut6_lut4_residual" \
  "$ROOT_DIR/experiments/san_francisco/run_lut6_lut4_residual.sh" "$@"
