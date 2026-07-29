# Current Technical Route

## Scope

As of 2026-07-29, the active training system contains Phase 1, Phase 2, and
the new LUT-as-neuron Phase 3. Retired LUT-as-operator, LUT5/LUT6, C8, and MLP
routes remain archived.

## Active Phases

### Phase 1

Phase 1 trains the full-precision complex Bi-Real residual network.
Residual-block activations use `ComplexReLU`, and their convolutions use
`ComplexConv2d`.

### Phase 2

Phase 2 keeps the same architecture and state-dict keys, then replaces
residual-block activations with `BinaryComplexActivation` and convolutions
with `BinaryComplexConv2d`. The stem remains full precision unless
`--binary-stem` is explicitly supplied.

Phase 2 normally initializes from a Phase 1 checkpoint. The loader requires
all trainable parameters to map and rejects a checkpoint whose recorded
phase is not Phase 1. `--train-from-scratch` is available as an explicit
alternative.

### Phase 3

Phase 3 replaces every residual main-path `BinaryComplexConv2d` with
`PairLUTNeuronConv2d`. The default starts from a Phase 2 checkpoint: for each
output channel, the Phase 2
complex kernel is flattened in `[input_channel, ky, kx]` order and grouped into
pairs. Each pair becomes one neuron:

- Inputs: two binary complex activations, ordered as
  `[x0_real, x0_imag, x1_real, x1_imag]`.
- Outputs: one real bit and one imaginary bit, represented by two independent
  16-entry truth tables with the same four inputs.
- Initialization: enumerate all 16 input states, evaluate the two fixed Phase 2
  binary complex weights, add the two products, and threshold each real or
  imaginary sum with `>= 0`.
- Odd tails: a missing second flattened position is represented by a fixed-low
  dummy input and a zero mathematical weight during initialization.
- Scale: the Phase 2 complex-weight alpha is retained as a fixed per-output
  buffer; hard LUT counts use `(count - pair_count / 2) * 4 * alpha`.

With `--train-from-scratch`, no checkpoint is loaded. Stem, projection,
BatchNorm, classifier, and other ordinary parameters keep their framework
random initialization. `--lut-init-mode normal` samples every real/imag LUT
entry independently from `Normal(0, LUT_LOGIT_INIT)`. The optional
`--lut-init-mode bimodal` uses a 50/50 mixture of `Normal(-1, 0.2)` and
`Normal(+1, 0.1)`. Both modes record the initial sign as the sign-diff baseline,
and use a fixed LUT output scale of 1.0.

After conversion, the residual main path has no spatial weight parameters.
The LUT logits are trainable and receive gradients through multilinear LUT
interpolation; activation bits keep an STE path to preceding layers. Stem,
projection, BatchNorm, and classifier parameters remain trainable.

The default schedule uses 160 soft epochs with geometric temperature annealing
from 0.5 to 10.0, followed by 40 epochs of gradual soft-to-hard forward
blending. Backpropagation always follows the soft table derivative. Only
fully-hard epochs may create `Bestmodel_phase3.pt`, and checkpoints include
explicit hard real/imag truth tables plus sign-difference diagnostics.

`--lut-training-mode real_compatible` is an isolated controlled-comparison
path. LUT entries are hard `0/1` from epoch 1, while
`(hard - logit).detach() + logit` gives every selected logit an identity STE.
It does not use tau or a soft-to-hard transition, so every epoch is deployable.
The dedicated launcher additionally selects bimodal initialization, Adam,
LR=0.01 for ordinary and LUT parameters, linear decay over 256 epochs, no
gradient clipping or weight decay, batch size 256, CIFAR-10 AutoAugment and
standard normalization, label smoothing 0.1, and all 50k training images.
The LUT neuron remains the complex 4-input/2-output design; this mode aligns
the optimization recipe, not the network topology.

The LUT execution backend is independently selectable with
`--lut-kernel-mode {auto,floating,binary}`. The default `auto` keeps the
multilinear floating kernel for the annealing route and selects the packed
binary CUDA kernel for `real_compatible`. Binary mode packs activation bits
into 32-bit words, performs an exact hard truth-table lookup, and uses the same
selected-entry LUT gradient and neighboring-entry input gradient as the
real-domain implementation. Its inputs are the `0/1` values produced by
`(binary_activation + 1) / 2` and are thresholded at `0.5`; combined with the
complex Bi-Real activation STE, this gives the same input-gradient scale as the
real network's `0/1` activation. Binary and floating backends are regression
tested for exact forward and backward equality at hard Boolean corners,
including channel counts that cross both 32-bit packing boundaries.

## Commands

Phase 1:

```bash
GPU_ID=0 WORKDIR=runs/phase1 ./run_phase1.sh
```

Phase 2 from the retained canonical Phase 1 model:

```bash
GPU_ID=1 \
WORKDIR=runs/phase2 \
CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase1.pt \
./run_phase2.sh
```

Phase 3 from the retained canonical Phase 2 model:

```bash
GPU_ID=2 \
WORKDIR=runs/phase3_pair_lut \
CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase2.pt \
./run_phase3.sh
```

Phase 3 overrides are explicit environment variables:

```bash
GPU_ID=2 LUT_LR=0.005 LUT_LOGIT_INIT=1.0 \
LUT_TAU_MIN=0.5 LUT_TAU_MAX=10 \
LUT_ANNEAL_EPOCHS=160 LUT_HARD_TRANSITION_EPOCHS=40 \
WORKDIR=runs/phase3_pair_lut_lr005 ./run_phase3.sh
```

Phase 3 from scratch with random LUT logits:

```bash
GPU_ID=2 TRAIN_FROM_SCRATCH=1 LUT_LOGIT_INIT=1.0 LR=0.01 \
WORKDIR=runs/phase3_pair_lut_scratch ./run_phase3.sh
```

Phase 3 real-compatible controlled comparison:

```bash
GPU_ID=1 \
WORKDIR=runs/phase3_pair_lut_real_compatible \
./run_phase3_real_compatible.sh
```

This wrapper defaults `LUT_KERNEL_MODE=binary`. Set
`LUT_KERNEL_MODE=floating` only for a backend-equivalence diagnostic.

This recipe deliberately trains on all 50k CIFAR-10 training images and uses
test accuracy for selection to match the referenced real LUT code. Keep its
result separate from validation-selected runs used for final reporting.

The launcher defaults the ordinary network LR to 0.01 in scratch mode and
0.001 in Phase 2 checkpoint mode. An explicit `LR` always overrides this.

Common overrides:

```bash
GPU_ID=0 \
WORKDIR=runs/phase2_cosine \
CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase1.pt \
NUM_EPOCHS=200 \
BATCH_SIZE=256 \
LR=0.01 \
SCHEDULE=cosine \
./run_phase2.sh
```

Use a distinct `WORKDIR` for concurrent runs. Phase-specific names prevent
different phases from overwriting each other in one workdir, but two
concurrent runs of the same phase must not share a workdir.

## Training Contract

- Default epochs: 200.
- Default batch size: 128 per process.
- Default optimizer: SGD with momentum 0.9.
- Default base learning rate: 0.1.
- Default schedule: five-epoch warmup to the supplied base learning rate,
  followed by the Bi-Real piecewise decay.
- Available schedules: `bireal`, `cosine`, `constant`, and `linear`.
- No schedule may exceed the externally supplied base learning rate.
- No hidden tuning mode changes epochs, batch size, optimizer, or LR.
- Phase 3 LUT parameters use an independent learning rate and schedule and no
  weight decay. Defaults are `LUT_LR=0.01` and a constant LUT schedule.
- Validation accuracy selects the best checkpoint. With
  `--no-validation`, test accuracy is the selection metric.

Each workdir receives:

- `chkpts/Bestmodel_phase1.pt`, `Bestmodel_phase2.pt`, or
  `Bestmodel_phase3.pt`
- `chkpts/Lastmodel_phase1.pt`, `Lastmodel_phase2.pt`, or
  `Lastmodel_phase3.pt`
- `phase_metrics.json`
- `phase{N}_{train,val,test}_{loss,acc}.txt`
- `logs/train.txt`
- `pixel_mean.pt`

## Retained Baseline

Canonical assets remain in `bi_workdir/`:

- Phase 1 best validation accuracy: 0.9092; test accuracy: 0.8948.
- Phase 2 best validation accuracy: 0.8310; test accuracy: 0.8221.
- `bi_workdir/chkpts/Bestmodel_phase1.pt`
- `bi_workdir/chkpts/Bestmodel_phase2.pt`

The rebuilt loader was verified to map all 258 tensors from the retained
Phase 1 checkpoint into the Phase 2 model without missing keys.

## LUT Building Blocks

The LUT implementation areas are:

- `complexPyTorch/complexLayers.py`
- `complexPyTorch/lut_backend.py`
- `lut_cuda/`

`PairLUTNeuronConv2d` is active only in Phase 3. The earlier
`ComplexLUTConv2d` and `LUTAwareComplexBinaryConv2d` building blocks remain
importable but are not instantiated by the active model route.

## Archive

All retired route entrypoints, flow modules, tests, scripts, reports,
checkpoints, logs, and run directories are under
`backup/route_reset_phase12_20260729/legacy_tree/`. The machine-readable
inventory is `archive_manifest.json`.

The old experiment history remains at:

`backup/route_reset_phase12_20260729/legacy_tree/PHASE4_LUT_EXPERIMENT_LOG.md`

## Verification

```bash
conda run -n lut_net python -m unittest discover -s tests -v
conda run -n lut_net python scripts/audit_active_route.py
```

Analyze a new run before adding conclusions to the experiment log:

```bash
python scripts/analyze_training_run.py runs/phase3_pair_lut
```
