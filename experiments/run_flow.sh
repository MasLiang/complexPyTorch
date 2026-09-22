#!/usr/bin/env bash
# Dataset-agnostic staged trainer. Dataset-specific flags are forwarded verbatim.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m experiments.run_flow "$@"
