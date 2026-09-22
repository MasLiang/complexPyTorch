# OpenSARShip Native SLC

This benchmark uses audited four-plane float TIFF chips from the original
OpenSARShip SLC release. The fixed mapping is
`[band0+j*band1, band2+j*band3] -> complex64 [2,H,W]` (VH/VV according to the
release metadata). The first protocol retains Cargo, Tanker, and Other Type;
rare classes are excluded rather than merged.

Chips are center-cropped or zero-padded to 128x128 without interpolation.
The loader assigns whole scenes to splits, then removes every MMSI that appears
in more than one provisional split. The resulting train/validation/test sets
are both scene-disjoint and MMSI-disjoint. Per-polarization normalization is
train-only positive RMS scaling, preserving complex phase.

```bash
GPU_ID=5 PYTHON_BIN=/home/jliangbr/miniconda3/envs/lut_net/bin/python \
  ./experiments/opensarship_slc/run_full_flow.sh
```

The first invocation decodes retained SLC chips from the nested archive into
an ignored local cache under `datasets_external/opensarship/processed/`.
