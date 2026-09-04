#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${RUN_ROOT:-"$ROOT_DIR/runs/san_francisco/multiseed"}
SEEDS_CSV=${SEEDS:-1,2}
GPUS_CSV=${GPUS:-4,3}
SPLIT_SEED=${SPLIT_SEED:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
REFERENCE_ROOT=${REFERENCE_ROOT:-"$ROOT_DIR/runs/san_francisco/full_flow"}

IFS=',' read -r -a seeds <<<"$SEEDS_CSV"
IFS=',' read -r -a gpus <<<"$GPUS_CSV"
if [[ "${#seeds[@]}" -ne "${#gpus[@]}" ]]; then
  echo "SEEDS and GPUS must contain the same number of comma-separated items" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
pids=()
roots=()
labels=()
for index in "${!seeds[@]}"; do
  seed=${seeds[$index]}
  gpu=${gpus[$index]}
  seed_root="$RUN_ROOT/seed_${seed}"
  mkdir -p "$seed_root"
  echo "Starting San Francisco seed $seed on GPU $gpu"
  (
    GPU_ID="$gpu" SEED="$seed" SPLIT_SEED="$SPLIT_SEED" \
    RUN_ROOT="$seed_root" PYTHON_BIN="$PYTHON_BIN" \
      "$ROOT_DIR/experiments/san_francisco/run_core_flow.sh"
  ) >"$seed_root/flow.log" 2>&1 &
  pid=$!
  echo "$pid" >"$seed_root/pid"
  pids+=("$pid")
  roots+=("$seed_root")
  labels+=("seed_${seed}")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one seed failed; inspect each seed's flow.log" >&2
  exit "$status"
fi

if [[ -f "$REFERENCE_ROOT/bireal_flow/lut6_residual/test_metrics.json" ]]; then
  roots=("$REFERENCE_ROOT" "${roots[@]}")
  labels=("seed_0" "${labels[@]}")
fi

"$PYTHON_BIN" -m experiments.san_francisco.summarize_multiseed \
  --flow-roots "${roots[@]}" \
  --labels "${labels[@]}" \
  --output "$RUN_ROOT/multiseed_summary.md"
