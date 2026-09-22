# CIFAR-10 Protocol

CIFAR-10 is configured by [`protocol.py`](protocol.py) and is executed through
the repository-wide staged engine:

```bash
GPU_ID=0 PYTHON_BIN=/home/jliangbr/miniconda3/envs/lut_net/bin/python \
  ./experiments/run_flow.sh \
  --dataset cifar10 --data-root data --workdir runs/cifar10/unified_seed0
```

The protocol supplies RGB data loading, the learned real-to-complex frontend,
CIFAR augmentation, and its historical optimizer recipe. The common engine
runs `shared_fp -> bireal -> lut4 -> lut6_residual`; the final LUT6 stage is
available when `--lut4-parameterization categorical` is used, which is the
default for the unified flow.

BiReal activations are signed (`{-1,+1}`).
`BinaryComplexBitActivation` converts them to exact LUT input bits (`{0,1}`),
so the categorical LUT4 and dominance-conditioned LUT6 use the same hardware
address semantics as San Francisco.

The root-level `run_phase1.sh`, `run_phase2.sh`, and `run_phase3.sh` launchers
remain for compatibility with historical CIFAR checkpoints and reports. New
cross-dataset experiments should use `experiments/run_flow.sh`.
