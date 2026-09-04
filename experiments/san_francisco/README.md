# San Francisco AIRSAR Final Experiment

This package classifies the San Francisco AIRSAR scene with six complex C3
channels (`C11`, `C22`, `C33`, `C12`, `C13`, `C23`) and five labelled classes.
The dataset decoder lives in `datasets/san_francisco.py`; it decodes the
official STK-MLC scene, constructs complex C3, and fits real/imaginary
normalization statistics using training coordinates only.

The retained protocol uses 11x11 complex patches and a spatial block split:
10% train blocks, 10% validation blocks, and the remainder test blocks. Patches
crossing a split boundary are excluded. Checkpoints are selected by validation
OA and evaluated once on test data.

## Retained Core Flow

The final comparison has five points:

1. `fp_baseline`: independent FP complex CNN baseline.
2. `bireal_fp`: full-precision operator in the shared BiReal topology.
3. `bireal`: binary complex weights and BiReal activation, initialized from `bireal_fp`.
4. `bireal_lut`: categorical PairLUT4 main branches compiled from the BiReal checkpoint.
5. `lut6_residual`: LUT4 base plus a zero-mean dominance-conditioned LUT6 residual.

The core flow intentionally stops at LUT6 residual. The discarded LUT4 common
correction and shortcut ablations remain in local `runs/` history only and are
not part of the final reported method.

Run one seed serially:

```bash
GPU_ID=0 SEED=1 SPLIT_SEED=0 \
  ./experiments/san_francisco/run_core_flow.sh
```

Run reproducibility seeds in parallel, holding the spatial split fixed:

```bash
PYTHON_BIN=/home/jliangbr/miniconda3/envs/lut_net/bin/python \
SEEDS=1,2 GPUS=4,3 SPLIT_SEED=0 \
RUN_ROOT=runs/san_francisco/multiseed \
  ./experiments/san_francisco/run_multiseed_core_flow.sh
```

`SEED` controls training randomness, while `SPLIT_SEED` controls only spatial
block assignment. Do not mix results from different split seeds in a single
mean/std table.

## Final Results

The completed fixed-split three-seed snapshot is versioned in
[`results/multiseed_core_flow.md`](results/multiseed_core_flow.md). The raw
data, checkpoints, histories, and prediction maps stay outside git under
`data/` and `runs/`; see [`results/README.md`](results/README.md) to regenerate
the result snapshot from local runs.

## Sanity Check

```bash
GPU_ID=0 ./experiments/san_francisco/run_baseline.sh --smoke-test
```
