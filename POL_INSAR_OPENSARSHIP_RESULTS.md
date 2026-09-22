# Pol-InSAR-Island and OpenSARShip Complex LUT Results

Status: seed-0 full flows are in progress or pending GPU scheduling. This file
is deliberately not populated with smoke-test metrics. Regenerate it only from
completed seed roots with:

```bash
python tools/summarize_paper_flows.py \
  --pol-root runs/pol_insar_island/reduced_t3/seed0 \
  --opensarship-root runs/opensarship_slc/full_flow/seed0
```

The generated report contains the validation-selected FP, BiReal, LUT4, and
LUT6 test metrics, relative OA/AA changes, and the protocol recorded in each
run root. Use `tools/analyze_lut_usage.py` and
`tools/validate_hard_lut_export.py` after a completed LUT6 run to append LUT
utilization and exact hard-table equivalence evidence.
