# Final San Francisco Results

This directory contains small, versioned result snapshots for the retained
San Francisco AIRSAR comparison. It deliberately excludes raw AIRSAR data,
training histories, checkpoints, prediction maps, and other generated run
artifacts; those remain local under `data/` and `runs/`.

`multiseed_core_flow.md` and `multiseed_core_flow.json` record the completed
fixed-spatial-split, three-seed comparison among the independent FP baseline,
shared-topology FP, BiReal, categorical LUT4, and LUT6 residual models.
Every metric is final test OA/AA from a validation-OA-selected checkpoint.

Regenerate the snapshot after a completed run with:

```bash
python -m experiments.san_francisco.summarize_multiseed \
  --flow-roots runs/san_francisco/full_flow \
               runs/san_francisco/multiseed/seed_1 \
               runs/san_francisco/multiseed/seed_2 \
  --labels seed_0 seed_1 seed_2 \
  --output experiments/san_francisco/results/multiseed_core_flow.md
```

The source-run paths embedded in the JSON are provenance metadata only; they
are intentionally not part of the repository.
