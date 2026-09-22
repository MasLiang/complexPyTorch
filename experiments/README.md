# Unified Experiment Flow

`experiments/run_flow.sh` is the maintained staged-training entry point. It owns
all common mechanics: seed control, data loaders, optimizer/schedule execution,
OA/AA metrics, best-checkpoint selection, and the checkpoint transitions
`shared_fp -> bireal -> lut4 -> lut6_residual`.

Each dataset owns only a `protocol.py` file that provides its data bundle,
dataset-specific model construction, and stage hyperparameters. This keeps the
complex BiReal/LUT arithmetic and all phase transitions identical across
datasets while preserving dataset-appropriate input geometry and evaluation
rules.

## Current Protocols

- [`cifar10/`](cifar10/README.md): RGB input with the learned real-to-complex
  frontend. By default it uses train/test selection and the established CIFAR
  augmentation and label-smoothing recipe.
- [`san_francisco/`](san_francisco/README.md): native six-channel complex C3
  patches with spatial train/validation/test splitting and validation-OA
  selection.
- [`pol_insar_island/`](pol_insar_island/README.md): official FP1/L T3 masks,
  a spatial validation subset derived only from official train labels, and a
  five-pixel cross-split guard band.
- [`opensarship_slc/`](opensarship_slc/README.md): genuine dual-polarization
  Sentinel-1 SLC chip classification with scene-disjoint splitting and removal
  of all MMSI that would otherwise cross split boundaries.

Both use `BinaryComplexResNet`, signed BiReal activations, exact `{0,1}` LUT
address bits, categorical LUT4 tables, and the optional dominance-conditioned
LUT6 residual. Dataset-dependent differences are declared in protocol files,
not duplicated in trainers.

## Run

Run the full chain for CIFAR-10:

```bash
GPU_ID=0 PYTHON_BIN=/home/jliangbr/miniconda3/envs/lut_net/bin/python \
  ./experiments/run_flow.sh \
  --dataset cifar10 --data-root data --workdir runs/cifar10/unified_seed0
```

Run the full chain for San Francisco AIRSAR:

```bash
GPU_ID=0 PYTHON_BIN=/home/jliangbr/miniconda3/envs/lut_net/bin/python \
  ./experiments/run_flow.sh \
  --dataset san_francisco --data-root data/san_francisco \
  --workdir runs/san_francisco/unified_seed0 --seed 0 --split-seed 0
```

Run the two new paper benchmarks:

```bash
GPU_ID=5 PYTHON_BIN=/home/jliangbr/miniconda3/envs/lut_net/bin/python \
  ./experiments/pol_insar_island/run_full_flow.sh

GPU_ID=5 PYTHON_BIN=/home/jliangbr/miniconda3/envs/lut_net/bin/python \
  ./experiments/opensarship_slc/run_full_flow.sh
```

`--stages lut6_residual` automatically includes its parents in the correct
order. `--smoke-test` runs one optimizer step per required stage and writes
minimal transition checkpoints, making it useful for a new protocol or a new
machine.

## Add a Dataset

1. Add `experiments/<dataset>/protocol.py`, subclassing
   `experiments.flow.contracts.DatasetProtocol`.
2. Implement `build_bundle`, `build_model`, and `stages`.
3. Register the protocol in `experiments/run_flow.py`.
4. Add a transition test matching `tests/test_unified_flow.py`.

Raw data, generated checkpoints, histories, and prediction maps remain under
ignored `data/` and `runs/` directories. Version the reusable loader,
protocol, analysis scripts, README, and compact result snapshots only.
