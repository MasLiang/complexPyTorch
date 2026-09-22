# Project Handoff: Complex BiReal and FPGA LUT Workflow

This document is the operational handoff for moving this repository to another
server and continuing the work with Codex. It records the active code path,
the assets that Git deliberately does not contain, validated experiment
results, and the immediate next experiments.

## 1. Repository State and Source of Truth

- Remote: `git@github.com:MasLiang/complexPyTorch.git`
- Active branch: `route/bireal-two-lutconv-clean`
- Last workflow commit before this handoff: `d855de9 Add unified complex LUT experiment workflows`
- The only known local untracked directory at handoff time is
  `experiments/flevoland/`. It is an archived note, not part of the maintained
  workflow, and must not be added accidentally.

Start a new-server checkout with:

```bash
git clone --branch route/bireal-two-lutconv-clean \
  git@github.com:MasLiang/complexPyTorch.git
cd complexPyTorch
git log -1 --oneline
```

The maintained cross-dataset route is:

```text
FP / topology FP -> complex BiReal -> categorical LUT4 -> residual LUT6
```

Use the shared launcher `experiments/run_flow.sh` and the dataset protocol in
`experiments/<dataset>/protocol.py` for new work. The root-level
`run_phase*.sh`, `training.py`, and documents such as
`CURRENT_TECHNICAL_ROUTE.md` are valuable for the historical CIFAR route, but
are not the authoritative interface for new cross-dataset experiments.

Read documents in this order:

1. `HANDOFF.md` (this file): migration and current status.
2. `experiments/README.md`: unified workflow interface.
3. `PAPER_METHOD_DRAFT_CN.md`: method, equations, and hardware mapping.
4. `EXPERIMENT_LOG.md`: chronological implementation and result evidence.
5. Dataset-specific `experiments/<dataset>/README.md` files.

`POL_INSAR_OPENSARSHIP_RESULTS.md` is an older scaffold and still describes
some runs as pending. For completed Pol-InSAR results, use the dated entries
in `EXPERIMENT_LOG.md` and the table below.

## 2. What Must Be Copied Separately

Git intentionally ignores generated and large assets:

```text
data/                  datasets_external/       runs/
logs/                  chkpts/                  bi_workdir/
backup/                reports/                 lut_cuda/*.so
```

This is code-only version control. Copy the datasets and any checkpoints that
you want to reuse before moving a run. Example commands, executed from the old
server, are:

```bash
# Copy external datasets. Keep the directory names unchanged.
rsync -aP /home/jliangbr/workspace/complexPyTorch/datasets_external/ \
  NEW_HOST:/path/to/complexPyTorch/datasets_external/

# Copy an entire finished flow, including protocol/history/metrics/checkpoints.
rsync -aP /home/jliangbr/workspace/complexPyTorch/runs/pol_insar_island/ \
  NEW_HOST:/path/to/complexPyTorch/runs/pol_insar_island/

# Copy historical CIFAR data/checkpoints only if those experiments are needed.
rsync -aP /home/jliangbr/workspace/complexPyTorch/data/ \
  NEW_HOST:/path/to/complexPyTorch/data/
rsync -aP /home/jliangbr/workspace/complexPyTorch/bi_workdir/ \
  NEW_HOST:/path/to/complexPyTorch/bi_workdir/
```

The most useful existing Pol-InSAR parent/checkpoints are under:

```text
runs/pol_insar_island/log_coherence_block16/seed0/lut4/best.pt
runs/pol_insar_island/log_coherence_block16/seed0/lut6_comparator_ablation_durable/ste/lut6_residual/best.pt
```

These are not in Git. Without them, rerun the parent stages instead of trying
to start `lut6_residual` alone.

## 3. Environment and Native CUDA Extensions

The old server used Python 3.10 in conda environment `lut_net`, with a
CUDA-enabled PyTorch build. The project package itself declares only `torch`,
but the active loaders and scripts also use `torchvision`, `numpy`, and, for
visualization only, `matplotlib`.

On a new server, create an environment whose PyTorch build matches the local
NVIDIA driver and CUDA toolkit. Then install the repository package and the
runtime dependencies, for example:

```bash
conda create -n lut_net python=3.10 -y
conda activate lut_net
# Install a CUDA-compatible torch/torchvision pair using the official PyTorch command for the new server.
pip install numpy matplotlib
pip install -e .
```

The LUT backend imports six locally compiled CUDA modules:

```text
lut_cuda_grouped              lut_cuda_grouped_binary
lut_conv_fp32_cuda            lut_conv_binary_cuda
lut_conv_bffb_cuda            lut_conv_bafw_cuda
```

Their `.so` files are ignored because they are Python, PyTorch, CUDA, and GPU
architecture specific. Rebuild them in `lut_cuda/` after the new PyTorch/CUDA
environment is working, then verify all imports before training:

```bash
cd lut_cuda
python setup.py build_ext --inplace
cd ..
python - <<'PY'
import lut_cuda_grouped, lut_cuda_grouped_binary
import lut_conv_fp32_cuda, lut_conv_binary_cuda
import lut_conv_bffb_cuda, lut_conv_bafw_cuda
import complexPyTorch.lut_backend
print("LUT CUDA imports OK")
PY
```

`lut_cuda/setup.py` contains the extension declarations. Do not casually edit
`lut_cuda/` or `complexPyTorch/lut_backend.py`: the current CUDA backend was
fixed and validated separately. If the one-command build does not produce all
six modules in a new setuptools/PyTorch version, treat it as an environment
porting issue, retain the exact source, build the missing declaration(s), and
rerun the import check before changing model code.

## 4. Active Method and Hard-Logic Contract

The active model keeps the complex BiReal residual topology, complex BN,
projection shortcuts, stem, and classifier. It replaces only main-branch
binary complex convolutions with `PairLUT4ComplexConv2d`.

1. BiReal activation emits signed real/imaginary values in `{-1, +1}`.
2. These are encoded exactly as LUT address bits
   `s_r=(sign(x_r)+1)/2`, `s_i=(sign(x_i)+1)/2`, hence addresses are `{0,1}`.
3. Two adjacent complex activations are grouped in channel-major order:
   `channel -> kernel_row -> kernel_col`.
4. LUT4 address: `[s_r0, s_i0, s_r1, s_i1]`. Each group has two one-bit
   tables, one for real and one for imaginary output; group outputs are summed
   and calibrated by post complex BN.
5. LUT4 uses a joint four-class parameterization (`00`, `01`, `10`, `11`) so
   real and imaginary output bits are optimized together. Training forward is
   hard argmax; softmax provides the straight-through backward proxy.
6. Residual LUT6 adds a phase/Gray comparator bit per input:
   `[s_r0, s_i0, p0, s_r1, s_i1, p1]`.
   LUT4 logits are expanded to all four `(p0,p1)` combinations, so LUT6 starts
   exactly equivalent to LUT4. A zero-mean categorical residual then
   conditionally refines the four slices while alpha ramps from 0.1 to 1.0.

For any fixed alpha, `argmax(base + alpha * residual)` is already a fully hard
and exportable LUT6 truth table. Alpha is therefore a training/selection
parameter, not an FPGA multiplier. `--select-any-residual-alpha` allows every
fixed-alpha epoch to compete by validation OA; this is correct hardware-wise,
but must use a validation split and test only once after selection.

The current Pol-InSAR STE comparator has a hard forward and an STE backward
surrogate. The `stop` mode also has hard forward but blocks gradients through
the comparator. The phase bit is not an additional runtime network; it is a
LUT input bit. Deployment exports only hard truth tables, never softmax,
continuous logits, residual tensors, or alpha arithmetic.

## 5. Unified Workflow and Commands

All maintained runs invoke:

```bash
GPU_ID=<gpu> PYTHON_BIN=/path/to/lut_net/bin/python \
  ./experiments/run_flow.sh --dataset <dataset> --data-root <root> --workdir <run-root>
```

`GPU_ID` becomes `CUDA_VISIBLE_DEVICES`. The engine saves each stage under
`<workdir>/<stage>/` with `best.pt`, `best_aa.pt`, `last.pt`, `history.json`,
`training.log`, and test metrics. A full stage list is inferred automatically.

| Dataset key | Input / task | Main data root | Stage sequence |
| --- | --- | --- | --- |
| `cifar10` | RGB, 10 classes; learned real-to-complex frontend | `data` | `shared_fp -> bireal -> lut4 -> lut6_residual` |
| `san_francisco` | six complex C3 channels, five classes | `data/san_francisco` | `independent_fp -> shared_fp -> bireal -> lut4 -> lut6_residual` |
| `pol_insar_island_t3` | six complex T3 channels, 12 classes | `datasets_external/pol_insar_island` | `fp -> bireal_fp -> bireal -> lut4 -> lut6_residual` |
| `opensarship_slc` | two complex SLC polarizations, three classes | `datasets_external/opensarship` | `fp -> bireal_fp -> bireal -> lut4 -> lut6_residual` |

Suggested new-server smoke check (one optimizer batch per needed stage):

```bash
GPU_ID=0 PYTHON_BIN="$CONDA_PREFIX/bin/python" \
  ./experiments/run_flow.sh --dataset pol_insar_island_t3 \
  --data-root datasets_external/pol_insar_island \
  --workdir runs/smoke/pol_seed0 --smoke-test \
  --representation log_coherence --spatial-block-size 16 --split-seed 0
```

For the validated final Pol-InSAR setup:

```bash
GPU_ID=0 PYTHON_BIN="$CONDA_PREFIX/bin/python" \
  ./experiments/run_flow.sh --dataset pol_insar_island_t3 \
  --data-root datasets_external/pol_insar_island \
  --workdir runs/pol_insar_island/log_coherence_block16/seed0 \
  --seed 0 --split-seed 0 --representation log_coherence \
  --patch-size 11 --spatial-block-size 16 --batch-size 128
```

To run LUT6 only from a copied LUT4 parent, use the dedicated matched
comparator ablation launcher. It deliberately skips all parents:

```bash
GPU_ID=0 MODE=ste PYTHON_BIN="$CONDA_PREFIX/bin/python" \
PARENT_WORKDIR=runs/pol_insar_island/log_coherence_block16/seed0 \
WORKDIR=runs/pol_insar_island/log_coherence_block16/seed0/lut6_ste_repeat \
  ./experiments/pol_insar_island/run_lut6_comparator_ablation.sh
```

Use `MODE=stop` for the strict stopped-gradient control. Do not reuse a parent
from a different representation, patch size, spatial block size, or split
seed.

For durable remote training, run in tmux rather than a transient interactive
process:

```bash
tmux new-session -d -s pol_lut6_ste \
  'cd /path/to/complexPyTorch && exec env GPU_ID=0 MODE=ste PYTHON_BIN=/path/to/lut_net/bin/python PARENT_WORKDIR=runs/pol_insar_island/log_coherence_block16/seed0 WORKDIR=runs/pol_insar_island/log_coherence_block16/seed0/lut6_ste_repeat ./experiments/pol_insar_island/run_lut6_comparator_ablation.sh > runs/pol_insar_island/log_coherence_block16/seed0/lut6_ste_repeat/driver.log 2>&1'
tmux attach -t pol_lut6_ste
```

## 6. Dataset Protocol Details

### CIFAR-10

This is historically important but not a strictly independent final protocol:
the legacy setup may select on test when no validation split exists. Keep its
numbers separate from spatial-validation PolSAR results.

### San Francisco AIRSAR

Native complex C3 `[C11,C22,C33,C12,C13,C23]`, 11x11 patches, and a spatial
block train/validation/test split. Best checkpoints select validation OA;
test is evaluated once after selection. Three-seed compact results live in
`experiments/san_francisco/results/`.

### Pol-InSAR-Island

Use the FP1/L upper-T3 block `[T11,T22,T33,T12,T13,T23]` as `complex64
[6,3616,2502]`. The healthy frozen representation is `log_coherence`:

```text
[log(T11), log(T22), log(T33),
 T12/sqrt(T11*T22), T13/sqrt(T11*T33), T23/sqrt(T22*T33)]
```

Protocol: official train/test masks; spatial validation tiles derived only
from official training labels; patch size 11; guard radius 5; block size 16;
split seed 0. Do not switch representation after entering BiReal/LUT stages.

### OpenSARShip SLC

Use native four-plane TIFF chips mapped as
`[band0 + j*band1, band2 + j*band3]`, center crop/pad to 128, and the
Cargo/Tanker/Other class protocol. The loader creates an ignored processed
cache on its first run. Its split is scene-disjoint and removes all MMSI that
would cross splits. The implementation is ready, but do not report a result
until a complete audited flow finishes.

## 7. Results at Handoff

### CIFAR-10 (historical, test-selected; not directly comparable to the spatial protocols)

| Model | Best test accuracy | Interpretation |
| --- | ---: | --- |
| FP Phase 1 | 89.48% | historical full precision reference |
| Complex BiReal | about 85.05% | binary convolution reference |
| Categorical LUT4 | 82.53% | LUT-neuron base |
| LUT6 residual continuation | 84.56% | residual improves over LUT4 |
| LUT6 with optional common correction | 85.09% | best historical result; not default across datasets |

### Pol-InSAR-Island (frozen log-coherence protocol, seed 0, validation-selected)

| Stage / model | Test OA | Test AA | Test macro-F1 |
| --- | ---: | ---: | ---: |
| FP | 0.8335 | 0.7326 | 0.7430 |
| BiReal-topology FP | 0.8494 | 0.7482 | 0.7656 |
| BiReal | 0.8537 | 0.7803 | 0.7872 |
| Categorical LUT4 | 0.8617 | 0.7807 | 0.7927 |
| Original LUT6 residual (alpha end only) | 0.8599 | 0.7753 | 0.7886 |
| LUT6 residual, stop comparator, any-alpha selection | 0.859701 | 0.763154 | 0.782175 |
| LUT6 residual, STE comparator, any-alpha selection | 0.862540 | 0.775765 | 0.790634 |

The STE run selected epoch 74 with fixed alpha 0.6521. It is better than stop
and has a small OA edge over LUT4 (+0.00081), but lower AA and macro-F1 than
LUT4. This is a single-seed observation, not a conclusive ranking.

### San Francisco AIRSAR (current reported snapshot)

| Model | Test OA | Test AA |
| --- | ---: | ---: |
| BiReal topology FP | 92.92% | 81.36% |
| BiReal | 94.35% | 79.99% |
| Categorical LUT4 | 93.55% | 78.87% |
| LUT6 residual | 93.74% | 78.97% |

LUT6 gave a small OA increase over LUT4 but not a comprehensive class-balanced
gain. The optional common correction did not transfer here; it stays an
ablation, not a default method.

## 8. Analysis and Verification Tools

Use scripts rather than manual log parsing:

```bash
# Summarize the residual-alpha candidates in a completed LUT6 stage.
python tools/analyze_residual_alpha.py \
  runs/pol_insar_island/log_coherence_block16/seed0/lut6_residual/history.json

# Check LUT4/LUT6 address use and dominance utilization.
python tools/analyze_lut_usage.py --help

# Verify that exported hard tables match model LUT outputs and final logits.
python tools/validate_hard_lut_export.py --help

# Build the compact paper-flow result table after complete runs.
python tools/summarize_paper_flows.py --help
```

The lightweight transition regression is in `tests/test_unified_flow.py`. The
old `lut_net` environment did not have `pytest`; it was verified by directly
calling the four test functions with temporary directories. Install pytest on
the new server if preferred, or run the direct harness from the experiment log.

After every source edit, append a concise entry to `EXPERIMENT_LOG.md`. For
analysis/report updates, add or extend a reusable script first where practical
instead of manually reprocessing logs. These are repository-level working
rules from `AGENTS.md`.

## 9. Current Interpretation and Recommended Next Work

1. Repeat LUT4 and LUT6-STE on multiple Pol-InSAR training/split seeds before
   presenting the 0.081 pp OA difference as a method win. Report OA, AA,
   macro-F1, and per-class deltas together.
2. Test a layer-wise residual-alpha schedule that is more conservative in deep
   stages. Current LUT6 conditional sensitivity falls with depth (about
   17.96%, 12.34%, and 9.54% in stages 2, 3, and 4 for final Pol STE).
3. Only after the seed study, test a PolSAR-specific third bit or a
   class-balanced objective as explicit ablations. Ordinary cross-entropy is
   currently the baseline objective.
4. Finish an audited OpenSARShip full flow before claiming cross-domain
   generality.
5. Do not revive the archived LUT-as-operator, magnitude-LUT5, C8, MLP/NIN,
   or TripleLUT6 routes without a new hypothesis. They are historical
   material, not the active baseline.

## 10. Migration Completion Checklist

1. Clone the branch and confirm `git status` is clean.
2. Create a CUDA-compatible `lut_net` environment and install the project.
3. Rebuild and import-check every native LUT extension.
4. Copy the required raw dataset directory and, if continuing a stage, its
   exact parent run directory.
5. Run one `--smoke-test` command on the new GPU.
6. Run a short/full stage in a new workdir; never overwrite a reference run.
7. Verify a completed LUT6 checkpoint with hard-table export validation.
8. Record the machine, environment versions, command, and outcome in
   `EXPERIMENT_LOG.md`.
