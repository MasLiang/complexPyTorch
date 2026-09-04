# San Francisco Multi-Seed Core-Flow Results

All rows report final test OA/AA from the checkpoint selected by validation OA. The intended reproducibility protocol fixes the spatial split and varies only the training seed.

## Per-seed test results

| Seed | Independent FP OA/AA | Shared FP OA/AA | BiReal OA/AA | LUT4 OA/AA | LUT6 residual OA/AA | LUT6-LUT4 OA pp |
|---|---:|---:|---:|---:|---:|---:|
| seed_0 | 94.67%/81.77% | 92.92%/81.36% | 94.35%/79.99% | 93.55%/78.87% | +0.94 |
| seed_1 | 94.26%/83.47% | 92.26%/75.97% | 93.55%/79.50% | 93.53%/78.21% | +0.94 |
| seed_2 | 93.57%/81.26% | 93.80%/81.89% | 92.84%/75.83% | 93.62%/79.63% | +0.93 |

## Mean and sample standard deviation

| Stage | OA | AA |
|---|---:|---:|
| fp_baseline | 94.17% +/- 0.56% | 82.17% +/- 1.15% |
| bireal_fp | 92.99% +/- 0.77% | 79.74% +/- 3.28% |
| bireal | 93.58% +/- 0.76% | 78.44% +/- 2.28% |
| lut4 | 93.57% +/- 0.05% | 78.91% +/- 0.71% |
| lut6 | 93.64% +/- 0.18% | 78.84% +/- 0.60% |
