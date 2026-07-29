#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATADIR=${DATADIR:-"$ROOT_DIR/data"}
CONDA_ENV=${CONDA_ENV:-lut_net}
GPU_ID=${GPU_ID:-0}
PHASE5_ANCHOR_SLICE=${PHASE5_ANCHOR_SLICE:-0}
PHASE5_TIES_ONLY=${PHASE5_TIES_ONLY:-1}
PHASE5_MAX_IMMEDIATE_VAL_DROP=${PHASE5_MAX_IMMEDIATE_VAL_DROP:-0.01}
PHASE5_MIN_CYCLE_VAL_GAIN=${PHASE5_MIN_CYCLE_VAL_GAIN:-0.0}

CHECKPOINT=${CHECKPOINT:-"$ROOT_DIR/bi_workdir/chkpts/Bestmodel_phase4.pt"}
WORKDIR=${WORKDIR:-"$ROOT_DIR/runs/phase5_lut5_anchor${PHASE5_ANCHOR_SLICE}_ties_c1d3"}

NUM_EPOCHS=${NUM_EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-128}
START_FILTER=${START_FILTER:-11}
NUM_BLOCKS=${NUM_BLOCKS:-3}
OPTIMIZER=${OPTIMIZER:-sgd}
LR=${LR:-0.001}
SCHEDULE=${SCHEDULE:-cosine}
MIN_LR_FACTOR=${MIN_LR_FACTOR:-0.1}
COMP_INIT=${COMP_INIT:-complex_independent}

LUT_SETS=${LUT_SETS:-1}
LUT_ALLOCATION=${LUT_ALLOCATION:-layer}
LUT_SETS_PER_CHANNEL=${LUT_SETS_PER_CHANNEL:-1}
LUT_LOGIT_INIT=${LUT_LOGIT_INIT:-2.0}
LUT_LR=${LUT_LR:-0.02}
LUT_SCHEDULE=${LUT_SCHEDULE:-follow_base}
LUT_SEARCH_EPOCHS=${LUT_SEARCH_EPOCHS:-40}
LUT_SEARCH_WEIGHT_EPOCHS_PER_LUT=${LUT_SEARCH_WEIGHT_EPOCHS_PER_LUT:-3}
LUT_SEARCH_MAX_FLIPS_PER_COMMIT=${LUT_SEARCH_MAX_FLIPS_PER_COMMIT:-1}
LUT_SEARCH_MAX_FLIPS_PER_LAYER=${LUT_SEARCH_MAX_FLIPS_PER_LAYER:-1}
LUT_SEARCH_FLIP_COOLDOWN=${LUT_SEARCH_FLIP_COOLDOWN:-2}
LUT_SEARCH_INIT_FLIP_PROB=${LUT_SEARCH_INIT_FLIP_PROB:-0.05}
LUT_SEARCH_SCORE_TEMPERATURE=${LUT_SEARCH_SCORE_TEMPERATURE:-0.25}
LUT_SEARCH_COMMIT_THRESHOLD=${LUT_SEARCH_COMMIT_THRESHOLD:-0.7}
LUT_SEARCH_SPARSITY=${LUT_SEARCH_SPARSITY:-0.0}
LUT_SEARCH_FREEZE_BN=${LUT_SEARCH_FREEZE_BN:-1}

if [[ "$PHASE5_ANCHOR_SLICE" != "0" && "$PHASE5_ANCHOR_SLICE" != "1" ]]; then
  echo "PHASE5_ANCHOR_SLICE must be 0 or 1" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Phase 4 checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if (( LUT_SEARCH_EPOCHS % (LUT_SEARCH_WEIGHT_EPOCHS_PER_LUT + 1) != 0 )); then
  echo "LUT_SEARCH_EPOCHS must contain complete 1+C/D recovery cycles" >&2
  exit 1
fi

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
  PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=(python)
fi

TIES_ARGS=()
if [[ "$PHASE5_TIES_ONLY" == "1" ]]; then
  TIES_ARGS+=(--phase5-ties-only)
fi

FREEZE_BN_ARGS=()
if [[ "$LUT_SEARCH_FREEZE_BN" == "1" ]]; then
  FREEZE_BN_ARGS+=(--lut-search-freeze-bn)
fi

echo "Running Phase 5 anchored LUT5 search"
echo "GPU: $GPU_ID"
echo "Checkpoint: $CHECKPOINT"
echo "Workdir: $WORKDIR"
echo "Anchor slice: d=$PHASE5_ANCHOR_SLICE"
echo "Trainable slice: d=$((1 - PHASE5_ANCHOR_SLICE))"
echo "Ties only: $PHASE5_TIES_ONLY"
echo "Cycle: 1 LUT score epoch + $LUT_SEARCH_WEIGHT_EPOCHS_PER_LUT weight recovery epoch(s)"
echo "Search/final epochs: $LUT_SEARCH_EPOCHS / $NUM_EPOCHS"
echo "LUT LR / threshold: $LUT_LR / $LUT_SEARCH_COMMIT_THRESHOLD"
echo "Max flips global/per-layer: $LUT_SEARCH_MAX_FLIPS_PER_COMMIT / $LUT_SEARCH_MAX_FLIPS_PER_LAYER"
echo "Immediate val drop / min cycle gain: $PHASE5_MAX_IMMEDIATE_VAL_DROP / $PHASE5_MIN_CYCLE_VAL_GAIN"

"${PYTHON_CMD[@]}" "$ROOT_DIR/training.py" \
  --phase 5 \
  --datadir "$DATADIR" \
  --workdir "$WORKDIR" \
  --checkpoint "$CHECKPOINT" \
  --num-epochs "$NUM_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --start-filter "$START_FILTER" \
  --num-blocks "$NUM_BLOCKS" \
  --spectral-pool-scheme "${SPECTRAL_POOL_SCHEME:-none}" \
  --comp_init "$COMP_INIT" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --schedule "$SCHEDULE" \
  --min-lr-factor "$MIN_LR_FACTOR" \
  --lut-inputs 5 \
  --lut-sets "$LUT_SETS" \
  --lut-allocation "$LUT_ALLOCATION" \
  --lut-sets-per-channel "$LUT_SETS_PER_CHANNEL" \
  --lut-logit-init "$LUT_LOGIT_INIT" \
  --lut-init-mode binary \
  --lut-sign-flip-prob 0 \
  --lut-lr "$LUT_LR" \
  --lut-schedule "$LUT_SCHEDULE" \
  --lut-search-epochs "$LUT_SEARCH_EPOCHS" \
  --lut-search-weight-epochs-per-lut "$LUT_SEARCH_WEIGHT_EPOCHS_PER_LUT" \
  --lut-search-max-flips-per-commit "$LUT_SEARCH_MAX_FLIPS_PER_COMMIT" \
  --lut-search-max-flips-per-layer "$LUT_SEARCH_MAX_FLIPS_PER_LAYER" \
  --lut-search-flip-cooldown "$LUT_SEARCH_FLIP_COOLDOWN" \
  --lut-search-init-flip-prob "$LUT_SEARCH_INIT_FLIP_PROB" \
  --lut-search-score-temperature "$LUT_SEARCH_SCORE_TEMPERATURE" \
  --lut-search-commit-threshold "$LUT_SEARCH_COMMIT_THRESHOLD" \
  --lut-search-sparsity "$LUT_SEARCH_SPARSITY" \
  --phase5-anchor-slice "$PHASE5_ANCHOR_SLICE" \
  --phase5-max-immediate-val-drop "$PHASE5_MAX_IMMEDIATE_VAL_DROP" \
  --phase5-min-cycle-val-gain "$PHASE5_MIN_CYCLE_VAL_GAIN" \
  "${TIES_ARGS[@]}" \
  "${FREEZE_BN_ARGS[@]}" \
  "$@"
