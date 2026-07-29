# Code Change Log

## 2026-07-29 - Restore archived LUT-as-operator route

This branch restores the last pre-reset LUT-as-operator source route from
`backup/route_reset_phase12_20260729`.

Restored core files:

- `training.py` from `legacy_tree` (`sha256 af8f142a...e5b84a3`).
- `complexPyTorch/complexBinaryResNet.py` from `legacy_tree`
  (`sha256 11878672...ca36f1`), because it contains the Phase
  1/2/2.1/3/3.1/3.5/3.6/4/5 router required by the archived trainer.
- `complexPyTorch/complexLayers.py` from the retained snapshot
  (`sha256 52fa269b...8f1a30`). It is byte-identical to
  `complexLayers_lut_as_op_backup.py`.
- `complexPyTorch/lut_backend.py` from the retained snapshot
  (`sha256 57b37956...6a77`). It is byte-identical to
  `lut_backend_lut_as_op_backup.py`.
- Archived flow modules, launchers, analysis scripts, and tests from their
  original repository-relative paths.

`complexBinaryResNet_lut_as_op_backup.py` was inspected but not copied over
the router: despite its filename, its contents explicitly support only Phase
1 and Phase 2, so it is incompatible with the archived multi-phase trainer.

Excluded from Git: checkpoints, run outputs, datasets, logs, Python bytecode,
CUDA shared objects/object files, and all other generated build artifacts.

Verification:

- All restored Python entrypoints pass `py_compile`.
- All restored shell launchers pass `bash -n`.
- The archived unit suite passes 106/106 tests.
