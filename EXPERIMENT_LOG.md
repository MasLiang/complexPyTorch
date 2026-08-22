<!-- experiment-entry:route-reset-phase12-20260729 -->
## 2026-07-29 - Phase 1/2 route reset and legacy-route archive

### 1. 本次路线切换的目标与结论

本次不是在旧实验树上继续修补，而是对活动代码做一次基线重置：

- 活动训练路线只保留 Phase 1 和 Phase 2。
- Phase 1 是 full-precision complex Bi-Real ResNet。
- Phase 2 是 binary complex activation + binary complex convolution weight。
- Phase 2 默认从 Phase 1 checkpoint 初始化，也可显式使用 `--train-from-scratch`。
- Phase 2.1、Phase 3/3.1/3.2/3.5/3.6、Phase 4/4.2/5，以及 LUT5、LUT6、C8、magnitude、dominance、MLP/NIN、fixed-analytic 等旧路线全部退出活动入口。
- 按用户要求，`complexPyTorch/complexLayers.py`、`complexPyTorch/lut_backend.py` 和 `lut_cuda/` 中的 LUT complex layer 与后端实现继续保留，但 Phase 1/2 模型不会实例化 LUT module。
- 旧代码、日志、报告、checkpoint 和运行结果均归档，不做删除。

### 2. 归档前仍在运行的任务

清理开始时发现 fixed-analytic dominance 任务仍在运行：

- Workdir: `runs/fixed_analytic_dominance_from_phase1_cmpbeta1_e200`
- 计划 epoch: 200
- epoch 174 时先写入 `route_reset_snapshot.md`
- 终止信号生效前，正在执行的 epoch 175 完整写入日志
- 最终状态：175/200，incomplete
- 最佳 validation accuracy：0.6250，epoch 173
- epoch 173 对应 test accuracy：0.6217
- 独立最佳 test accuracy：0.6242，epoch 166
- epoch 175：train 0.6031，validation 0.6164，test 0.6187
- epoch 174 快照统计：8/8 code occupied，dominance ratio 0.4982，occupancy entropy 0.9857，三个 bit 的记录梯度均非零

任务已通过进程组 SIGTERM 停止，并确认没有旧 trainer/launcher 进程继续运行。最终摘要由
`scripts/analyze_training_run.py` 生成到
`backup/route_reset_phase12_20260729/fixed_analytic_final_summary.md`。

### 3. 主归档内容

归档根目录：

`backup/route_reset_phase12_20260729/`

机器可读清单：

`backup/route_reset_phase12_20260729/archive_manifest.json`

旧目录树：

`backup/route_reset_phase12_20260729/legacy_tree/`

主归档工具移动了以下 42 个 repository-relative path，并在 manifest 中记录目标路径、文件数量、总字节数；小于 16 MiB 的单文件还记录 SHA-256：

1. `training.py`
2. `training.py.orig`
3. `training_direct_lut5.py`
4. `training_dual_lut6.py`
5. `training_fixed_analytic.py`
6. `training_mlp_flow.py`
7. `complexPyTorch/complexBinaryResNet.py`
8. `complexPyTorch/complexBinaryResNet.py.orig`
9. `complexPyTorch/complexLayers.py.orig`
10. `complexPyTorch/directLut5Flow.py`
11. `complexPyTorch/dualLut6Flow.py`
12. `complexPyTorch/fixedAnalyticDominanceFlow.py`
13. `complexPyTorch/mlpFlow.py`
14. `run_all_phases.py`
15. `run_direct_lut5.sh`
16. `run_dual_lut6.sh`
17. `run_fixed_analytic.sh`
18. `run_magnitude_lut5.sh`
19. `run_mlp_flow.sh`
20. `run_phase2p1.sh`
21. `run_phase3p1.sh`
22. `run_phase3p5.sh`
23. `run_phase3p6.sh`
24. `run_phase4.sh`
25. `run_phase5.sh`
26. `PHASE4_LUT_EXPERIMENT_LOG.md`
27. `scripts/`：34 个旧脚本
28. `tests/`：37 个旧测试文件
29. `reports/`：22 个旧报告文件
30. `runs/`：655 个旧运行文件
31. `logs/`：2 个根日志文件
32. `chkpts/`：20 个根 checkpoint 文件
33. `bi_workdir/`：旧完整工作目录
34. `train_acc.txt`
35. `train_loss.txt`
36. `val_acc.txt`
37. `val_loss.txt`
38. `test_acc.txt`
39. `test_loss.txt`
40. `pixel_mean.pt`
41. `__pycache__/`
42. `complexPyTorch/__pycache__/`

主归档完成后的静态扫描又发现 `build/` 是旧源码的 generated package 副本，仍含旧 Phase 3/4/5 内容。因此：

- `archive_legacy_routes.py` 增加可复用的 `--supplement` 机制。
- `build/` 被作为 supplement 移至 `legacy_tree/build/`。
- supplement 同样写入 `archive_manifest.json`。
- 活动树不再存在这份陈旧 generated source。

目录型记录内部的每个文件均按原相对路径原样移动；完整逐文件定位以 manifest 和 `legacy_tree/` 为准。

### 4. 归档支持文件

- `backup/route_reset_phase12_20260729/archive_legacy_routes.py`
  - dry-run-first 的归档工具。
  - 支持首次归档和后续 `--supplement`。
  - 新增路径规范化，拒绝绝对路径和 `..` 越界。
- `backup/route_reset_phase12_20260729/archive_manifest.json`
  - 记录 42 个主路径、retained snapshots、恢复的 Phase 1/2 资产和 `build/` supplement。
- `backup/route_reset_phase12_20260729/README.md`
  - 说明归档结构、旧任务最终状态、重要路径和恢复方法。
- `backup/route_reset_phase12_20260729/restore_legacy_path.py`
  - 可选择单文件或目录恢复。
  - 默认 dry run。
  - 只复制，不删除归档内容。
  - 目标存在时拒绝覆盖。
- `backup/route_reset_phase12_20260729/fixed_analytic_final_summary.md`
  - 由统一日志分析脚本生成的中断任务最终摘要。
- `backup/route_reset_phase12_20260729/legacy_tree/runs/fixed_analytic_dominance_from_phase1_cmpbeta1_e200/route_reset_snapshot.md`
  - epoch 174 时的停止前现场记录。
- `retained_snapshots/`
  - 保存重置前的 `README.md`、`CHANGELOG.md`、`run_training.sh`、`complexLayers.py` 和 `lut_backend.py` 快照。

此前已有的 `backup/magnitude_lut5_20260714/` 保持原样，没有覆盖或搬动。

### 5. 保留并恢复的 Phase 1/2 资产

从归档的旧 `bi_workdir/` 中复制回活动树：

- `bi_workdir/chkpts/Bestmodel_phase1.pt`
- `bi_workdir/chkpts/Bestmodel_phase2.pt`
- `bi_workdir/pixel_mean.pt`
- `bi_workdir/phase_metrics.json`

`phase_metrics.json` 已过滤为只含 `phase1` 和 `phase2`。

保留基线：

- Phase 1：best validation 0.9092，test 0.8948，epoch 170
- Phase 2：best validation 0.8310，test 0.8221，epoch 126

### 6. 活动代码逐文件变更

- `complexPyTorch/complexBinaryResNet.py`
  - 从旧多路线 router 重建为 Phase 1/2-only model。
  - `ACTIVE_PHASES=(1, 2)`。
  - Phase 1 block 使用 `ComplexReLU + ComplexConv2d`。
  - Phase 2 block 使用 `BinaryComplexActivation + BinaryComplexConv2d`。
  - Phase 1/2 保持相同 state-dict key 和 tensor shape。
  - Phase 3+ 构造会直接报错。
  - 删除活动模型对 C8、MLP、LUT5/LUT6 和实验 flow module 的依赖。

- `training.py`
  - 从旧 4800+ 行多路线 trainer 重建为 Phase 1/2-only trainer。
  - CLI 的 `--phase` choices 只有 1 和 2。
  - 删除 LUT/C8/MLP/Phase 3+ 参数与调度入口。
  - 删除 `bireal_tune` 隐式参数覆盖。
  - `batch_size`、`num_epochs`、`lr`、optimizer 和 schedule 始终服从外部参数。
  - schedule 只保留 `bireal`、`cosine`、`constant`。
  - warmup 只增长到用户设置的 base LR，不会放大到另一个固定值。
  - cosine 对数值误差做上限钳制，严格不超过 base LR。
  - Phase 2 默认要求 Phase 1 checkpoint；支持显式 `--checkpoint` 和 `--train-from-scratch`。
  - checkpoint loader 去除 `module.`/`_orig_mod.` wrapper，按 key 和 shape 映射，并要求所有 trainable parameters 成功加载。
  - 每个 epoch 原子保存 `Lastmodel_phaseN.pt`。
  - best 保存为 `Bestmodel_phaseN.pt`，不会再使用跨 phase 的共享名字。
  - history 文件也增加 phase 前缀。
  - DDP rank 0 使用 unwrapped model 做验证，避免单 rank 进入 DDP forward 同步。
  - compile 请求仍支持；complex operator 下将 inductor 路由到 `aot_eager`。
  - CIFAR10/CIFAR100/SVHN、validation split、no-validation、pixel mean 和 augmentation 基线保留。

- `run_training.sh`
  - 保留为通用 Phase 1/2 launcher。
  - 默认 dataset/workdir 改为 repository-relative 明确目录。
  - DDP 统一使用选定 `PYTHON_BIN -m torch.distributed.run`。
  - compile 默认关闭。
  - 使用 `exec` 让退出码和信号正确传递。

- `run_phase1.sh`
  - 新增 Phase 1 专用入口。
  - 支持 `GPU_ID`、`WORKDIR`、`DATADIR`、`NUM_EPOCHS`、`BATCH_SIZE`、`LR`、`SCHEDULE`、`WEIGHT_DECAY` 等环境变量。
  - 可选显式 Phase 1 warm-start checkpoint。

- `run_phase2.sh`
  - 新增 Phase 2 专用入口。
  - 默认 checkpoint 为 `bi_workdir/chkpts/Bestmodel_phase1.pt`。
  - 支持 `CHECKPOINT` 覆盖和 `TRAIN_FROM_SCRATCH=1`。
  - 支持 binary weight scale 配置。
  - 默认 workdir 与 Phase 1 分离。

- `scripts/analyze_training_run.py`
  - 新增统一训练日志分析器。
  - 解析 epoch train/validation/test 指标、配置、完成状态、best validation、best test 和 last。
  - 同一个日志含多次 invocation 时，只分析最后一次运行。
  - 支持 Markdown/JSON 和 `--output`。

- `scripts/append_experiment_log.py`
  - 新增统一日志追加工具。
  - 支持 body/body-file/stdin、日期和 dedupe key。
  - 本条记录即通过该脚本写入。

- `scripts/audit_active_route.py`
  - 新增活动路线审计。
  - 检查训练 phase 集合、活动 phase launcher、实验 module 是否退出活动树。
  - 检查 Phase 1/2 model 不实例化 LUT。
  - 同时检查保留的 LUT layer 仍可导入。

- `scripts/__init__.py`
  - 将新维护脚本目录定义为可测试的 Python package。

- `tests/test_phase12_route.py`
  - 覆盖 Phase choices、实验 CLI 清理、model layer 类型、state key 一致性、checkpoint 全量映射、Phase 1/2 前后向、LR 上限和 LUT layer 保留。

- `tests/test_log_tools.py`
  - 覆盖最新 invocation 日志解析和实验日志 dedupe。

- `tests/__init__.py`
  - 定义当前测试 package。

- `README.md`
  - 首页增加当前 Phase 1/2 路线、运行命令、归档入口和 LUT layer 保留说明。

- `CURRENT_TECHNICAL_ROUTE.md`
  - 新增当前技术路线契约。
  - 记录 phase 定义、转换规则、默认参数、输出文件、并发 workdir 约束、基线、归档和验证命令。

- `EXPERIMENT_LOG.md`
  - 新建活动路线实验总日志。
  - 旧完整历史保存在归档的 `PHASE4_LUT_EXPERIMENT_LOG.md`，不再继续向旧日志追加。

### 7. 明确保留且本次未改写的实现

以下文件在进入清理前已有用户/前序实验改动。本次没有回退、没有重写，只按要求继续保留：

- `complexPyTorch/complexLayers.py`
- `complexPyTorch/lut_backend.py`
- `lut_cuda/` 下的 source、binary 和 build artifacts

这些文件仍可能包含 LUT4/LUT5/LUT6/C8 等实现，但它们现在只是未接入活动 Phase 1/2 的 building blocks。活动 router、trainer 和 launcher 均没有相应入口。

导入 LUT backend 时会出现 PyTorch 关于 `torch.cuda.amp.custom_fwd/custom_bwd` 的 FutureWarning。该警告来自保留代码，不影响本次 Phase 1/2 路线审计，因此本次未顺手改动。

### 8. 验证结果

在 `lut_net` conda 环境中完成：

- `python -m py_compile`：新的 trainer、model、分析脚本、审计脚本、归档与恢复脚本全部通过。
- `bash -n`：`run_training.sh`、`run_phase1.sh`、`run_phase2.sh` 全部通过。
- `python -m unittest discover -s tests -v`：10/10 tests passed。
- Phase 1 和 Phase 2 model state-dict key 集合完全相同。
- retained canonical Phase 1 checkpoint 含 258 个 tensor；258/258 全部映射到新的 Phase 2 model，0 unmatched、0 shape mismatch、0 missing buffer。
- canonical model Phase 1/2 参数量均为 261,719。
- 小模型 Phase 1/2 forward 和 backward 均成功。
- route audit：
  - active route exposes only Phase 1/2
  - Phase 1/2 models instantiate no LUT modules
  - retained LUT complex layers remain importable
- selective restore tool dry-run 成功，未写入目标。
- reusable analyzer 成功读取旧中断任务并识别 incomplete。
- 进程扫描确认没有 retired experiment trainer/launcher 继续运行。
- 验证过程生成的活动 `__pycache__` 已清除；历史 cache 仍保存在 archive 中。

本次没有重新执行 200 epoch 的完整 Phase 1/2 训练；完整训练成本较高，兼容性由 canonical checkpoint 全量映射、前后向测试和已有基线指标覆盖。下一次真正训练后，应先运行 `scripts/analyze_training_run.py`，再基于生成摘要追加实验结论。

### 9. 新路线的运行与分析规则

Phase 1：

```bash
GPU_ID=0 WORKDIR=runs/phase1 ./run_phase1.sh
```

Phase 2：

```bash
GPU_ID=1 \
WORKDIR=runs/phase2 \
CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase1.pt \
./run_phase2.sh
```

并发实验必须使用不同 `WORKDIR`。Phase-specific 文件名可避免 Phase 1 与 Phase 2 互相覆盖，但两个相同 phase 的并发任务如果共用同一 workdir，仍会覆盖各自的 best/last checkpoint。

后续任何新技术路线应作为新的、显式命名的 flow 加入，并先与当前 Phase 1/2 baseline 对比；不要重新把旧实验分支塞回 Phase 1/2 router。每次修改文件均继续追加本日志，每次分析新 run 先调用统一分析脚本。

<!-- experiment-entry:route-reset-phase12-final-verification -->
## 2026-07-29 - Route-reset final verification supplement

### 补充文件变更

- `.gitignore`
  - 新增 Python `__pycache__`、`*.pyc`、pytest/coverage cache 和 `*.egg-info` 忽略规则。
  - 新增仅针对 repository root 的 `/build/`、`/runs/`、`/logs/`、`/chkpts/` 忽略规则。
  - 不忽略 `backup/`、`bi_workdir/`、`*.pt`、LUT CUDA source 或现有二进制，因此归档与保留基线仍清晰可见。
  - 目的：完成本次提交后，运行测试和训练不会再次用 generated cache/build output 污染活动源码状态。

- `backup/route_reset_phase12_20260729/archive_legacy_routes.py` 与
  `restore_legacy_path.py`
  - 已设置 executable bit；所有 root launcher 和三个维护脚本也均确认可执行。

### 最终补充验证

- Phase 1 launcher 使用 `PYTHON_BIN=echo` dry-run：
  - `GPU_ID=7`
  - `NUM_EPOCHS=3`
  - `BATCH_SIZE=16`
  - `LR=0.007`
  - `SCHEDULE=constant`
  - 所有值原样进入 `training.py`，未发生覆盖。
- Phase 2 launcher 使用 `PYTHON_BIN=echo` dry-run：
  - `GPU_ID=6`
  - `CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase1.pt`
  - `NUM_EPOCHS=4`
  - `BATCH_SIZE=32`
  - `LR=0.009`
  - `SCHEDULE=cosine`
  - phase、checkpoint 与全部数值原样进入 `training.py`。
- 最终 manifest JSON 可解析：
  - primary moved paths：42
  - supplement：`build/`，6 files，81,857 bytes
  - restored Phase 1/2 assets：4
- 活动 entrypoint retired-route 关键词扫描无结果。
- 除 `legacy_tree/` 中有意保留的历史 cache 外，活动树没有 `__pycache__`。
- 本次修改文件的 `git diff --check` 通过。
- 全 worktree `git diff --check` 仍报告
  `lut_cuda/lut_conv_binary_cuda_backend.cu` 第 290、327 行的既有 trailing whitespace；该文件属于用户此前保留的 LUT CUDA 改动，本次没有修改或回退。
- 最终进程扫描无旧训练任务。

<!-- experiment-entry:pair-lut-neuron-phase3-implementation -->
## 2026-07-29 - LUT-as-neuron Phase 3 implementation

### 1. 路线目标

本次在保留 Phase 1/2 和旧 LUT building blocks 的前提下，新增全新的 Phase 3：从“LUT as operator + spatial weight 协同优化”切换为“LUT as neuron”。

每个输出 channel 的 Phase 2 二值复数卷积核按 `[input_channel, ky, kx]` flatten，每两个复数权重组成一个 neuron。该 neuron 接收两个二值复数 activation，即 4 个输入 bit `[x0_r, x0_i, x1_r, x1_i]`，并通过共享输入的两张 4-LUT 分别输出 real/imag 1 bit。每个 output channel 和每个 flatten pair 都有独立、可学习的 real/imag truth table。

### 2. Phase 2 到 Phase 3 初始化

对每一对固定的 Phase 2 二值复数权重，遍历全部 16 种输入状态，直接计算：

```text
sum_r = x0_r*w0_r - x0_i*w0_i + x1_r*w1_r - x1_i*w1_i
sum_i = x0_r*w0_i + x0_i*w0_r + x1_r*w1_i + x1_i*w1_r
lut_r = 1[sum_r >= 0]
lut_i = 1[sum_i >= 0]
```

Phase 2 latent weight 先按原 `complex_binary_weight` 语义取 sign；其 complex magnitude mean alpha 作为 Phase 3 固定 `output_scale` buffer。硬 LUT 输出按 `(count - pair_count/2) * 4 * alpha` 恢复为对称累加尺度。

当 `C * K * K` 为奇数时，最后一组的第二个复数输入连接固定-low dummy bit，初始化时第二个数学权重为 0。因此这一尾组只有 4 个可达输入状态，不额外引入 spatial weight。

canonical `bi_workdir/chkpts/Bestmodel_phase2.pt` 转换结果：

- 18 个 residual main-path conv 全部转换成功。
- 1934 个 flatten pair LUT neuron。
- 7 层存在 odd tail。
- 2,022,592 个 real+imag LUT entries。
- source weight exact-zero component 为 0。
- 初始化后 sign diff 为 0。
- Phase 3 主干中没有 `conv_r/conv_i` spatial weight 参数。
- Phase 3 总参数量为 2,031,663；增加来自独立 LUT logits，不是保留旧主卷积权重。

### 3. 可微 LUT 与退火

LUT logits 通过 `(tanh(logit * tau) + 1) / 2` 得到 soft entry。渐进 hard 使用：

```text
forward = soft + hard_ratio * stop_gradient(hard - soft)
hard = 1[logit >= 0]
```

因此 forward 从 soft 连续混合到 hard，而 backward 在包括 `hard_ratio=1` 的 fully-hard 阶段仍沿 soft derivative 更新 LUT。输入 activation 在 forward 使用二值 bit，backward 使用 STE 回传到前层。

默认 200 epoch：

- epoch 1-160：`hard_ratio=0`，tau 从 0.5 几何退火到 10。
- epoch 161-200：tau 固定 10，`hard_ratio` 从 1/40 逐步增加到 1。
- epoch 200 是第一个 fully-hard training/evaluation epoch。
- Phase 3 的 epoch 数若不足以到达 fully hard，会在训练开始前拒绝运行。
- 只有 fully-hard epoch 可以更新 `Bestmodel_phase3.pt`；`Lastmodel_phase3.pt` 每个 epoch 正常保存。
- checkpoint 额外保存显式 uint8 hard real/imag truth tables、初始化转换报告和 LUT sign-diff。
- 每 10 epoch及 hard 阶段记录 tau、hard ratio、real/imag sign diff。

LUT 参数使用独立 optimizer group，无 weight decay。默认 network LR 为 0.001 cosine，LUT LR 为 0.01 constant；两者均可从外部配置。

### 4. 文件修改总结

- `complexPyTorch/complexLayers.py`
  - 新增 `annealed_binary_table`。
  - 新增 `PairLUTNeuronConv2d`，包含 Phase2 truth-table 初始化、pair connection、odd-tail dummy、CPU reference、CUDA K=4 backend、soft/hard STE、sign diff 和 hard table 导出。
  - 保留原有 `ComplexLUTConv2d`、`LUTAwareComplexBinaryConv2d` 等旧 building blocks。

- `complexPyTorch/complexBinaryResNet.py`
  - active phases 更新为 1/2/3。
  - Phase 3 residual main conv 路由到 `PairLUTNeuronConv2d`。
  - Phase 1/2 路由保持 full precision / binary；直接构造模型时默认根据 phase 推导 `is_binary`。
  - 增加 LUT logit/tau 构造参数，不恢复旧实验 flow 参数。

- `training.py`
  - 新增严格的 Phase2-to-Phase3 converter。
  - 新增 LUT optimizer group、独立 LR schedule、tau/hard-ratio scheduler、sign-diff 聚合和 hard table capture。
  - Phase 3 禁止 from scratch，要求 Phase 2 checkpoint。
  - 只允许 fully-hard 指标选择 best checkpoint。
  - 增加全部 Phase 3 CLI 参数及启动前校验。

- `run_phase3.sh`
  - 新增可执行 launcher。
  - 默认 canonical Phase2 checkpoint、200 epochs、160+40 annealing、network cosine LR 与 LUT constant LR。
  - 所有关键配置均可由环境变量覆盖。

- `tests/test_phase12_route.py`
  - active route 断言更新为 Phase 1/2/3。
  - 保留 Phase1/2 state-key 和 checkpoint 兼容测试。
  - 新增 Phase3 类型及“无 spatial weight”断言。

- `tests/test_pair_lut_phase3.py`
  - 新增 16 状态 truth-table exact test。
  - 新增 odd-tail dummy independence test。
  - 新增 soft 与 fully-hard STE 下输入/LUT 双向梯度测试。
  - 新增 Phase2 checkpoint 全层转换和默认退火终点测试。

- `scripts/audit_active_route.py`
  - 路线审计更新为仅暴露 Phase 1/2/3。
  - 验证 Phase1/2 无 LUT，Phase3 仅使用 pair-LUT 且无主干 spatial weight，旧 LUT building blocks 仍可导入。

- `README.md`
  - 增加 Phase3 路线说明与启动命令。

- `CURRENT_TECHNICAL_ROUTE.md`
  - 详细记录 pair-LUT 数学定义、flatten 顺序、odd-tail、尺度、退火、checkpoint 和命令契约。

- `EXPERIMENT_LOG.md`
  - 通过 `scripts/append_experiment_log.py` 追加本条，作为后续 Phase3 实验分析基线。

### 5. 验证结果

- Python 语法检查通过。
- `run_phase3.sh`、`run_phase2.sh`、`run_training.sh` 的 `bash -n` 通过。
- launcher dry-run 确认所有默认参数原样进入 `training.py`。
- exhaustive 16-state truth table 与直接两次复数乘加+阈值结果完全一致，最大误差 0。
- canonical full model 在 GPU 上 forward/backward 成功，输出 shape `(2, 10)` 且 finite。
- 36 个 real/imag LUT parameter tensor 全部获得梯度，LUT gradient sum 约 1071.24。
- CPU reference 与 CUDA K=4 kernel 对比：最大绝对误差约 `5.86e-6`，平均绝对误差约 `9.02e-7`。
- 完整单元测试 15/15 passed。
- active route audit passed。

本次没有启动 200 epoch CIFAR-10 正式训练，因此尚无 Phase 3 accuracy 结论。第一组正式结果出来后，应先运行：

```bash
python scripts/analyze_training_run.py runs/phase3_pair_lut
```

再基于本条中的 sign diff、soft/hard transition 和 hard checkpoint 指标分析 LUT neuron 是否真正改变 truth table并带来收益。

<!-- experiment-entry:pair-lut-phase3-random-scratch -->
## 2026-07-29 - Phase 3 random initialization from scratch

### 目标与行为

按新实验要求，Phase 3 现在允许完全不读取 Phase 1/2 checkpoint，直接随机初始化整网训练。默认 checkpoint truth-table 初始化仍保留，只有显式设置 `--train-from-scratch` 或 launcher 环境变量 `TRAIN_FROM_SCRATCH=1` 时才使用新路径。

scratch 模式下：

- stem、projection、BN affine、classifier 等普通参数使用 PyTorch 原生随机初始化。
- 每个 pair-LUT neuron 的 real/imag entry logits 独立采样自 `Normal(0, LUT_LOGIT_INIT)`。
- 默认 `LUT_LOGIT_INIT=1.0`，此时它表示随机分布标准差，不是 Phase2 truth-table 模式下固定的绝对 logit magnitude。
- 初始 hard truth table 由随机 logit sign 决定，理论上 0/1 各约 50%。
- 随机初始 sign 被保存为 sign-diff baseline，因此 epoch 0 的 sign diff 为 0，后续统计仍表示相对本次随机起点发生的真值表变化。
- 没有 Phase2 alpha 时，固定 `output_scale=1.0`。
- soft-to-hard 的 160+40 默认退火、独立 LUT LR、hard-only best checkpoint 规则不变。
- checkpoint 中使用 `phase3_initialization.mode=random_normal` 记录该实验来源，不会与 Phase2 truth-table 初始化混淆。

### 文件修改

- `complexPyTorch/complexLayers.py`
  - `PairLUTNeuronConv2d` 新增 `initialize_random(logit_std)`。
  - 随机初始化同步设置 logits、初始 sign、output scale 和 initialized buffer。
  - Phase2 初始化报告增加 `mode=phase2_truth_table`，使两条路径可区分。

- `training.py`
  - 新增 `initialize_phase3_random`，遍历并初始化全部 pair-LUT 层。
  - Phase3 的 `--train-from-scratch` 不再被禁止。
  - 无 checkpoint 时必须显式初始化 LUT，避免零 logits 或未初始化 layer。
  - checkpoint metadata 统一改为 `phase3_initialization`。

- `run_phase3.sh`
  - 新增 `TRAIN_FROM_SCRATCH=1` 分支。
  - scratch 分支只传 `--train-from-scratch`，不再传 `--checkpoint`。
  - 启动信息打印 random-normal 标准差，避免误认实验来源。

- `tests/test_pair_lut_phase3.py`
  - 新增随机分布均值/标准差、初始化状态、sign-diff baseline 和前后向测试。
  - 新增 Phase3 scratch checkpoint resolver 测试。

- `README.md`
  - 增加随机从头训练的最短命令。

- `CURRENT_TECHNICAL_ROUTE.md`
  - 将 Phase3 定义更新为 Phase2 truth-table 与 random-normal 两种显式初始化模式。
  - 记录 scratch 参数和 output scale 语义。

- `EXPERIMENT_LOG.md`
  - 使用 `scripts/append_experiment_log.py` 追加本条。

### 建议首轮命令

```bash
GPU_ID=0 \
TRAIN_FROM_SCRATCH=1 \
LUT_LOGIT_INIT=1.0 \
LUT_LR=0.01 \
LR=0.001 \
WORKDIR=runs/phase3_pair_lut_scratch_std1 \
./run_phase3.sh
```

这是结构变化较大的实验：不再继承 Phase2 的 82% 左右起点，早期准确率明显更低属于预期。判断价值时应重点观察训练是否持续上升、LUT sign diff 是否增长、以及最终 fully-hard epoch 是否恢复，而不能用最初几个 epoch 直接与 checkpoint fine-tuning 比较。

<!-- experiment-entry:pair-lut-phase3-random-scratch-verification -->
## 2026-07-29 - Phase 3 random scratch verification supplement

### LR 补充

完整从头训练不再沿用 checkpoint fine-tuning 的 network LR。run_phase3.sh 现在按模式选择默认值：

- TRAIN_FROM_SCRATCH=1：network LR=0.01。
- Phase2 checkpoint 初始化：network LR=0.001。
- 两种模式的 LUT LR 默认均为 0.01 constant。
- 外部显式设置 LR 时始终覆盖上述默认值。

因此 scratch 首轮正式命令以 LR=0.01 为准；上一条记录命令中的 LR=0.001 已被本补充替代。

### 最终验证

- canonical-size 随机 Phase3：18 层、1934 个 pair-LUT neuron 全部成功初始化。
- GPU full-model batch-2 forward/backward 成功，输出 shape (2, 10) 且全部 finite。
- LUT gradient sum 约 281.15。
- 完整测试 17/17 passed。
- active route audit passed。
- scratch launcher dry-run 确认参数包含 --train-from-scratch，不包含 --checkpoint，network LR 为 0.01，LUT LR 为 0.01。
- 本次相关文件 git diff --check 通过。

最终建议命令：

```bash
GPU_ID=0 \
TRAIN_FROM_SCRATCH=1 \
LUT_LOGIT_INIT=1.0 \
LUT_LR=0.01 \
LR=0.01 \
WORKDIR=runs/phase3_pair_lut_scratch_std1 \
./run_phase3.sh
```

<!-- experiment-entry:real-versus-complex-pair-lut-comparison -->
## 2026-07-29 - Real LUT-BiReal versus complex pair-LUT comparison

### 对比对象

读取并对照了实数工程 ~/workspace/LutNet/cifar10_bireal 中的 train.py、lut_birealnet.py 和实际存在的 lut_layer_main_compile.py，并使用新脚本解析 log_2x.txt、log_2x_baseline.txt、log_2x_baseline_3values.txt、log_lr0p01.txt。用户提到的 lut_layer_main_compiler.py 实际文件名是 lut_layer_main_compile.py。

### 实数 LUT-BiReal 的关键训练机制

- 每个 residual block 使用 BinaryActivation -> lut_conv_group -> BatchNorm -> residual add。
- activation forward 是 0/1 hard bit，backward 是 Bi-Real piecewise surrogate。
- 每个 LUT 默认 6 输入、1 输出，64-entry；一个 3x3 patch flatten 后每 6 个 scalar bit 分组，各 LUT 输出求和。
- LUT logits 使用双峰初始化：50% 来自 Normal(-1, 0.2)，50% 来自 Normal(+1, 0.1)。这使初始 hard table 约 50/50，但 entry 几乎都远离 0。
- BasicBlock 构造后立即 hard.fill_(1)，所以从第一个 epoch 开始 forward 就是物理 0/1 truth table。
- hard LUT backward 使用 (hard - logits).detach() + logits，对 logit 的代理导数恒等于 1；没有 tanh 饱和。
- optimizer 是 Adam，LUT 和其他参数共用 LR；默认无 weight decay。
- 没有 clip_grad_norm 或 clip_grad_value。
- LR 使用线性衰减；训练 256 epoch。
- 使用全部 50k CIFAR-10 train、标准 mean/std normalize、RandomCrop、flip、CIFAR10 AutoAugment 和 label smoothing 0.1。

### 历史曲线

可复用脚本 scripts/analyze_real_lut_bireal_log.py 的结果：

- log_2x.txt（LUT）：epoch 1/5/10/20/50/100/150/256 为 23.02/34.68/41.76/48.83/69.34/76.59/80.39/83.47%，best 83.47%。
- log_lr0p01.txt（LUT，未完整结束）：epoch 1/5/10/20/50/100 为 23.17/34.60/46.63/55.46/74.79/85.06%，epoch 132 best 87.50%。
- log_2x_baseline.txt：best 81.45%。

当前复数 random scratch run runs/phase3_pair_lut_scratch_std1 在 epoch 20：train 31.78%，val 34.66%，test 33.76%。它并非完全没有学习，但明显慢于实数 LUT 同期 48.83%-55.46%。

### 最关键差异与诊断

1. 优化器不一致。
   实数 LUT 使用 Adam；当前复数 run 使用 SGD。LUT 有 2,022,592 个独立 entry，单 batch 对 entry 的更新具有稀疏、尺度不均特性，Adam 的 per-entry 自适应步长比 SGD 更匹配。

2. 全局 gradient clipping 严重压缩 LUT 更新。
   当前复数训练默认 clipnorm=1、clipval=1，实数版本完全不裁剪。canonical-size batch-16 诊断得到：LUT raw L2=1.769，其他参数 L2=23.878，全局 L2=23.944，因此 clip coefficient 仅 0.04176。LUT mean |grad|=3.28e-4，经 clipping 和 SGD LR=0.01 后，典型单步更新约 1.37e-7。这个量级无法让 Normal(0,1) logits 有效跨 0。

3. 当前 LUT 的离散 truth table 几乎没有学习。
   epoch 1 sign diff 29/2,022,592；epoch 11 仅 278/2,022,592，即 0.0137%。准确率上升主要可能来自 stem、projection、BN 和 classifier 对近似随机 LUT feature 的适配，而不是 LUT operation 本身改变。

4. hard/soft 和代理梯度完全不同。
   实数版本从 epoch 1 就 hard-forward，部署目标与训练 forward 一致，同时 logit backward 恒等于 1。当前复数版本前 160 epoch soft-forward，之后才渐进 hard，并且即使 fully hard，backward 仍是 tanh soft derivative。tau 增大后，远离 0 的 logits 会梯度饱和。这会重现此前“LUT sign 不动”的问题。

5. 初始化不同。
   实数版本是稳定的 bimodal ±1 初始化；当前复数是 Normal(0,1)，约 38% entry 落在 |logit|<0.5。若切 hard identity STE，后者可能产生过多早期翻转；若保持 soft+tanh，又会形成长期连续 surrogate 与最终 hard table 的目标错位。

6. 数据和训练 recipe 也更强。
   实数版本有 Adam、256 epoch、全 50k train、AutoAugment、标准差 normalize、label smoothing；当前复数是 45k train、200 epoch、较弱 augmentation、无 label smoothing。这能解释部分 generalization gap，但不能解释 sign diff 几乎为 0；LUT 更新机制仍是第一优先级。

### 判断与下一步优先级

当前 run 可以继续用于证明 soft-SGD 路径的上限，但不适合作为“随机 LUT 能否从头训练”的公平结论。最优先应做一个 real-compatible controlled branch，而不是先改复数数学结构：

1. LUT 和普通参数改用 Adam，先用统一 LR=0.001 或对 LUT 单独 0.001。
2. 关闭 global clipnorm/clipval，或至少将 LUT 排除在普通参数的全局裁剪之外。
3. 增加 bimodal random init：Normal(-1,0.2)/Normal(+1,0.1)。
4. 增加 hard-forward + identity-logit-STE 模式，第一 epoch 就使用 hard table，不做 tau annealing。
5. 保持当前 4-input/2-output complex pair-LUT 结构不变，先隔离训练 recipe 的影响。
6. 记录每 epoch LUT grad norm、update norm、sign diff；若这条 real-compatible 模式仍失败，再检查复数双输出、ComplexBN 或 pair grouping 的结构影响。

建议下一组实验与当前 run 并行，而不是立刻覆盖：一组严格复现实数 recipe（Adam + no clipping + bimodal + hard identity STE），一组只改 Adam + no clipping，以做消融。

### 文件修改

- scripts/analyze_real_lut_bireal_log.py：新增可复用实数 LUT 日志解析器，输出 epoch 数、best/last、LR 首尾和固定 milestone accuracy。
- EXPERIMENT_LOG.md：使用 scripts/append_experiment_log.py 追加本次对比分析。

本次只做分析和日志工具，没有修改当前正在运行的 Phase3 数学或训练行为，也没有停止现有 run。

<!-- experiment-entry:phase3-real-compatible-20260729 -->
## 2026-07-29 - Phase3 real-compatible 训练机制对照分支

### 动机与可检验假设

对照 ~/workspace/LutNet/cifar10_bireal 后，当前复数 pair-LUT 从头训练较慢不能直接归因于复数 4-input/2-output 结构。现有 run 同时存在 SGD、全局 clipnorm=1、Normal(0,1)、长 soft annealing 等配方差异。一次 canonical-size 梯度诊断中，LUT raw L2=1.769、其他参数 L2=23.878、全局 L2=23.944，因此全局裁剪系数约为 0.04176；SGD LR=0.01 下典型 LUT entry 更新只有约 1.37e-7。epoch 11 的 sign diff 也只有 278/2,022,592（0.0137%）。

本次新增隔离的 real-compatible 分支。假设是：如果训练不动主要来自优化配方，那么 hard-forward identity STE + Adam + no clipping 应显著增加 LUT sign diff，并加快前 20-50 epoch 的准确率增长；如果 sign diff 已明显增加但准确率仍低，下一步才应优先检查复数双输出、pair grouping、ComplexBN 或累加尺度等结构差异。

### 对照模式的数学行为

- 复数结构保持不变：每个 pair-LUT neuron 仍接收两个二值复数 activation，即 4 bit，共享地址并分别输出 real/imag 两个 bit。
- LUT 前向从 epoch 1 起严格 hard：entry 为 1[logit >= 0]。
- LUT 反向使用 identity-logit STE：(hard - logit).detach() + logit，因此 LUT entry 对自身 logit 的局部导数恒为 1，不再经过 tanh 饱和。
- real-compatible 模式不使用 tau annealing 或 hard transition；所有 epoch 都满足硬件可部署 forward，可参与 best checkpoint 选择。
- scratch LUT 初始化为 50/50 双峰混合：N(-1,0.2) 与 N(+1,0.1)。原 Normal(0,std) 与原 annealing 模式保留为默认，不受影响。

### 专用训练配方

run_phase3_real_compatible.sh 默认参数：

- train from scratch，256 epochs，单卡全局 batch size 256；
- Adam；普通参数 LR=0.01，LUT LR=0.01；
- 两组 LR 都使用从 base LR 开始、逐 epoch 线性下降的 schedule；
- weight decay=0，clipnorm=0，clipval=0；
- CIFAR-10 RandomCrop + horizontal flip + AutoAugment，标准 mean/std normalize；
- label smoothing=0.1；
- 使用全部 50k train，不设 validation，以复现实数代码的数据协议。

命令：

```bash
GPU_ID=1 \
WORKDIR=runs/phase3_pair_lut_real_compatible \
./run_phase3_real_compatible.sh
```

注意：no-validation 模式使用 test accuracy 选择 best checkpoint，只适合训练机制复现实验，不能作为无偏最终 test 报告。真正架构比较仍应另跑保留 validation 的版本。

### 观察指标

1. 每 10 epoch 的 Pair-LUT Sign Diff 数量与比例，必须明显高于 annealing+SGD run 的 0.0137%（epoch 11）。
2. epoch 10/20/50 accuracy 与实数 LUT 日志 milestone 46.63%/55.46%/74.79% 对照，但只作为优化速度参考，因为网络拓扑和复数双输出不同。
3. hard forward 从 epoch 1 开始，因此不存在 soft accuracy 与部署 hard accuracy 的转换落差。
4. 若 sign diff 仍接近 0，检查 Adam 实际 update norm 和地址 occupancy；若 sign diff 大量变化但 accuracy 剧烈波动，优先降低 LUT_LR，而不是重新引入全局 clipping。

### 文件修改

- complexPyTorch/complexLayers.py：为 PairLUTNeuronConv2d 增加 real_compatible hard identity STE；增加可配置双峰初始化；保留原 annealed_binary_table 默认行为。
- complexPyTorch/complexBinaryResNet.py：将 lut_training_mode 从模型入口传递到所有 Phase3 pair-LUT 层。
- training.py：增加 LUT init/training mode 参数、双峰配置、linear 普通/LUT schedule、label smoothing、real_lut CIFAR-10 augmentation；real-compatible 从 epoch 1 标记 fully hard；scratch initializer 记录具体初始化元数据。
- run_phase3.sh：暴露 optimizer、clipping、augmentation、label smoothing、LUT training/init mode 和双峰参数；支持 NO_VALIDATION。
- run_phase3_real_compatible.sh：新增一键对齐实数 LUT-BiReal 配方的独立启动器。
- tests/test_pair_lut_phase3.py：新增 hard 0/1 forward、identity gradient、双峰统计、首轮 fully-hard 和 linear LR 测试。
- tests/test_phase12_route.py：将 linear schedule 纳入不超过 base LR 的回归检查。
- scripts/audit_active_route.py：将新的合法 Phase3 recipe wrapper 纳入 active-route 白名单。
- README.md：增加对照模式命令、行为和 test-selection 风险说明。
- CURRENT_TECHNICAL_ROUTE.md：记录 normal/bimodal 初始化、anneal/real-compatible 两条 Phase3 训练路径及完整配方。
- EXPERIMENT_LOG.md：通过 scripts/append_experiment_log.py 追加本条详细记录。

### 验证

- conda run -n lut_net python -m unittest discover -s tests -v：21/21 通过。
- conda run -n lut_net python scripts/audit_active_route.py：通过。
- bash -n run_phase3.sh run_phase3_real_compatible.sh：通过。
- 使用 PYTHON_BIN=/bin/echo dry run 核对，完整参数包含 Adam、LR/LUT_LR=0.01、linear、no clipping、bimodal、real_lut、label smoothing=0.1、no-validation 和 train-from-scratch。
- git diff --check 的唯一告警来自用户此前保留的 lut_cuda/lut_conv_binary_cuda_backend.cu 两处尾随空格，本次没有修改该文件。

<!-- experiment-entry:git-route-binary-cuda-alignment-20260729 -->
## 2026-07-29 - Git 路线分支与 binary CUDA 全面对齐

### 路线分支

本次将两条技术路线正式分离，Git 中只提交代码、启动脚本、测试和必要文档，不提交 checkpoint、数据集、runs、训练日志或 CUDA 编译产物。

- 旧 LUT-as-operator 路线恢复到 `archive/lut-operator-legacy`，提交为 `bab0ade`，并已推送到 origin。恢复源来自 `backup/route_reset_phase12_20260729`；其中旧版 complexBinaryResNet 支持 Phase 1/2/2.1/3/3.1/3.5/3.6/4/5。
- 用户保留的 `complexLayers_lut_as_op_backup.py` 与 `lut_backend_lut_as_op_backup.py` 和路线重置快照完全一致。名为 `complexBinaryResNet_lut_as_op_backup.py` 的文件实际只允许 Phase 1/2，因此旧分支采用备份目录中真正支持旧 phase 的版本。
- 当前 LUT-as-neuron 路线位于 `route/lut-neuron-real-aligned`。它保留 Phase 1/2，并以两个二值复数输入、一个二值复数输出的 4-input/2-output pair-LUT 作为 Phase 3。

旧分支验证：Python 编译通过，全部 shell 启动器语法通过，旧路线单元测试 106/106 通过。提交前检查确认没有 `.pt`、`.pth`、`.so`、`.o`、`.pyc`、runs、data 或 chkpts 文件进入 Git。

### 与实数 LUT-BiReal 的 kernel 对齐

实数参考实现的真实调用链是 `train.py -> lut_birealnet.BasicBlock -> lut_conv_group -> lut_conv_func_group -> lut_kernel_group -> LUT6Function.apply -> torch.ops.mylib.lut_forward -> lut_cuda_grouped_binary.forward/backward`。其前向把浮点形式的 0/1 输入按 `>0.5` 硬索引；反向对 LUT 返回被选 entry 的梯度，对输入返回相邻两个 entry 的差值梯度。

当前复数 pair-LUT 的 real-compatible 路径现已使用同构的 packed binary CUDA lookup：

1. 两个复数 activation 形成 `[x0_r,x0_i,x1_r,x1_i]` 四个地址 bit，real/imag 使用相同地址、两张独立 16-entry table。
2. `BinaryComplexActivation` 先产生 -1/+1，再用 `(x+1)/2` 转成 0/1 probability 编码。binary kernel 使用 `>0.5`，避免把 0 错当成 high。
3. Bi-Real signed activation 的代理导数为 `2(1-|x|)`，随后的除以 2 恰好得到 `1-|x|`，与实数参考实现的 0/1 Bi-Real 反向尺度一致。
4. 输入 bit 按 channel 每 32 位打包，offset 覆盖 kernel 空间和输入 channel，shift 指定字内 bit。输出通道也由 CUDA kernel 分 tile 处理。
5. annealing 路线仍使用 floating multilinear kernel；`real_compatible` 默认使用 binary kernel。可用 `--lut-kernel-mode auto|floating|binary` 显式控制，且 binary 模式只允许 hard real-compatible forward。
6. CUDA 回归测试使用 18 个复数输入 channel 和 35 个输出 channel，覆盖多 input packed word 和多 output tile；binary/floating 在 hard Boolean corner 的 forward、activation gradient、real LUT gradient、imag LUT gradient逐项完全相等。

### 当前路线文件修改

- `complexPyTorch/lut_backend.py`：binary autograd wrapper 增加 signed/probability 输入编码；旧调用默认 signed，pair-LUT 使用 probability。
- `complexPyTorch/complexLayers.py`：PairLUTNeuronConv2d 增加 auto/floating/binary kernel 选择、packed offsets/shifts 构造和 binary CUDA 调用；阻止 annealing 错用 binary backend。
- `complexPyTorch/complexBinaryResNet.py`：把 LUT kernel mode 从模型入口传到所有 Phase3 block。
- `training.py`：增加 `--lut-kernel-mode` 并传入模型。
- `run_phase3.sh`：暴露、打印并传递 `LUT_KERNEL_MODE`。
- `run_phase3_real_compatible.sh`：默认 `LUT_KERNEL_MODE=binary`。
- `tests/test_pair_lut_phase3.py`：增加非法模式检查以及跨 packing/tile 边界的 CUDA forward/backward 精确等价测试。
- `README.md`：说明 real-compatible 默认 binary CUDA 与 0/1 梯度约定。
- `CURRENT_TECHNICAL_ROUTE.md`：记录 backend 选择、数学梯度关系和验证范围。
- `EXPERIMENT_LOG.md`：使用可复用的 `scripts/append_experiment_log.py` 写入本条记录。

### 下一次实验

使用 `run_phase3_real_compatible.sh` 从头训练，默认 Adam、普通参数/LUT LR 都为 0.01、linear 256 epochs、batch size 256、bimodal LUT logits、hard identity STE、binary CUDA、无全局梯度裁剪、50k train/no-validation。首先观察前 10-20 epoch 的准确率、LUT sign diff、LUT update norm 和 GPU 吞吐，再判断训练不动是否仍来自 complex pair-LUT 结构本身。

<!-- experiment-entry:binary-cuda-validation-handoff-20260729 -->
## 2026-07-29 - binary CUDA 回归验证与运行交接

### 验证结果

- `CUDA_VISIBLE_DEVICES=0 conda run -n lut_net python -m unittest discover -s tests -v`：23/23 通过。
- CUDA binary/floating 等价测试覆盖 18 complex input channels 和 35 output channels，forward 与全部输入/LUT gradient 完全一致。
- `conda run -n lut_net python scripts/audit_active_route.py`：通过，活动路线只暴露 Phase 1/2/3。
- `bash -n run_phase1.sh run_phase2.sh run_phase3.sh run_phase3_real_compatible.sh`：通过。
- real-compatible dry run 确认参数包含 `real_compatible`、`binary`、Adam、LR/LUT_LR=0.01、linear、bimodal、no clipping、batch size 256、no-validation 和 train-from-scratch。
- 1 epoch 前台诊断成功：训练进入 binary CUDA forward/backward，第 1 epoch test accuracy 25.84%，LUT sign diff 1350/2,022,592，说明 LUT entry 已发生更新；该诊断输出与 checkpoint 仅位于本地 `runs/debug_phase3_real_binary`，不进入 Git。
- 曾启动的 GPU 0 tmux 正式任务已按用户要求停止，GPU 由用户自行启动；推荐独立 workdir 为 `runs/phase3_pair_lut_real_aligned_binary_clean`。

### 用户运行命令

```bash
GPU_ID=0 \
WORKDIR=runs/phase3_pair_lut_real_aligned_binary_clean \
./run_phase3_real_compatible.sh
```

### 文件记录

本条仅通过 `scripts/append_experiment_log.py` 更新 `EXPERIMENT_LOG.md`，记录验证和运行交接；没有修改模型数学、训练配置或 CUDA 实现。

<!-- experiment-entry:phase3-pair-lut-stall-reference-diff-20260730 -->
## 2026-07-30 - Phase3 pair-LUT 109 epoch 停滞与实数参考差异复核

### 当前 run 的事实

使用 `scripts/analyze_training_run.py` 预处理 `runs/phase3_pair_lut_real_aligned_binary_clean`。分析时任务仍在运行，已完成 109/256 epoch：

- best test 49.12%（epoch 92）；
- epoch 109：train 40.40%，test 49.04%；
- epoch 20/50/75/100 test 分别为 47.86%/43.91%/48.07%/46.69%，已经形成长期平台，不是单纯收敛较慢；
- LUT sign diff 从 epoch 1 的 0.0603% 增长到 epoch 20 的 8.2916%、epoch 50 的 15.0882%、epoch 100 的 20.0208%、epoch 104 的 20.2569%。

因此“训不动”不是 LUT 参数没有更新。约五分之一 truth-table entry 已越过零点，但这些大量布尔变化没有形成更好的表示；继续降低 LR 的剩余 epoch 不太可能从 49% 自动跃迁到 80% 以上。

### 0.02 的来源复核

旧路线中表现较好的复数实验 `phase2p1_c8_octants_semantic_phase_beta2_bireal_lr01` 的真实 invocation 是：

```
--optimizer sgd --lr 0.1 --schedule bireal --batch-size 128
```

旧 bireal schedule 使用 5 epoch warmup，所以 epoch 1 打印 0.02，之后为 0.04、0.06、0.08，并从 epoch 5 起达到配置的 base LR 0.1。故记忆中的“初始 LR=0.02”是 SGD base LR=0.1 的首个 warmup 值，不是 Adam 固定 LR、LUT LR，也不是 bimodal logit 初始化尺度。

实数 LUT-BiReal 的最好现存日志 `log_lr0p01.txt` 明确从 Adam LR=0.01 开始：epoch 1/5/10/20/50/100 为 23.17%/34.60%/46.63%/55.46%/74.79%/85.06%，best 87.50%。当前 run 已经正确复现了这个 0.01 Adam 配方。因此直接将当前 Adam LR 和 LUT LR 都翻倍到 0.02 不是“修正遗漏”，只是新的超参数实验；鉴于当前已有 20% sign flip，它也可能加剧无序翻转。

### 已经对齐的部分

- Adam，betas 0.9/0.999；
- network/LUT 初始 LR 0.01，256 epoch 线性下降；
- weight decay 0、无 gradient clipping；
- batch size 256，对应每 epoch 196 batch；
- CIFAR-10 RandomCrop、horizontal flip、AutoAugment、标准 mean/std；
- label smoothing 0.1，全部 50k train、test selection；
- bimodal logits：50/50 的 N(-1,0.2) 和 N(+1,0.1)；
- hard forward + identity logit STE；
- binary CUDA 对 LUT entry 和 activation 使用 neighboring-entry difference gradient。

scheduler 只有 epoch 编号上的一个极小边界差异，不足以解释 35 个百分点的差距。weight decay 为 0 时，两边参数分组差异也不产生实际影响。

### 尚未对齐且可能重要的部分

1. **LUT 函数阶数不同**：实数实现每个 LUT 是 6 independent scalar bits -> 1 bit，有 64-entry 任意布尔函数；当前是两个完整复数 activation，即 4 correlated scalar bits -> real/imag 各 1 bit，每张表只有 16 entries。当前模型只能在固定两复数组内表达四变量交互，无法复现实数 LUT6 的六变量高阶交互。这是最大的表达能力差异。
2. **Block 顺序不同**：实数 block 是 `BinaryActivation -> LUT -> BatchNorm -> residual add`；当前是 `ComplexBatchNorm(pre) -> BinaryComplexActivation -> pair-LUT -> ComplexBatchNorm(post) -> residual add`。额外 pre-BN 会改变每层 truth-table 地址分布。
3. **BN 数学不同**：实数使用普通标量 BatchNorm（eps 1e-5，gamma 初始 1）；当前 post-BN 做 real/imag 2x2 协方差白化和交叉仿射，eps 1e-4，仿射对角初始 sqrt(2)。随机 hard truth table 不断翻转时，复数协方差统计和 LUT 同时移动，优化面明显更复杂。
4. **下采样 shortcut 不同**：实数为 AvgPool2d(stride=2) 后接 stride-1 1x1 conv；当前直接使用 stride-2 complex 1x1 conv。两者不是等价采样。
5. **输入和 stem 不同**：实数 RGB 直接进入实数 conv；当前先由 LearnImagBlock 生成虚部，再经全精度 complex stem。最后分类器也拼接 real/imag。这个结构保留了复数模型本身，但不能称为完全复制实数网络。
6. **尾部 padding 不同**：实数 LUT6 分组不足时重复输入 bit；当前奇数个 complex spatial position 时使用固定 low dummy bit。影响范围较小，但仍是差异。
7. **优化历史不同**：旧成功复数模型通常从 Phase1 checkpoint 开始，用 SGD+bireal schedule、batch 128；当前直接从随机 hard pair-LUT 开始，用 Adam+linear、batch 256。当前既不是旧复数训练路径，也只对齐了实数实现的训练 recipe，而没有对齐实数 LUT6 拓扑。

### 判断与下一步优先级

当前 run 可以停止；继续到 256 epoch 的信息增益很低。下一步不要先把所有 LR 机械改成 0.02，而应先回答“结构有能力、只是 scratch 难，还是 pair-LUT 压缩本身损失太大”：

1. **首选诊断**：从最佳 Phase2 checkpoint 枚举初始化 pair-LUT，第一 epoch 就 hard-forward，然后观察转换后的 epoch-0/1 accuracy。如果一开始接近 Phase2，说明 pair-LUT 表达能力够，问题主要在随机 scratch 优化；如果立即降到很低，说明两复数压成一复数的局部表示本身是瓶颈。
2. **优化器消融**：若 checkpoint 初始化可行，再给当前 K4 pair-LUT 增加与旧复数训练完全一致的 LUT-follow-base bireal schedule，即 SGD base LR 0.1、前五 epoch 0.02->0.1、batch 128。当前代码的 LUT schedule 尚不支持 bireal/follow-base，不能只靠现有环境变量严格复现。
3. **结构消融**：保持 K4/2-output 硬件约束，单独提供 real-style block：去掉 bn_pre、post 改为 real/imag 独立 BN、shortcut 使用 AvgPool+1x1。与当前 covariance-BN block 并行比较，避免把 LR 和拓扑同时改变。
4. Adam LR=0.02 可以作为低优先级对照，但不应作为主实验；当前 sign flip 已经足够多，主要矛盾不是 entry 无法越零。

<!-- experiment-entry:lut6-compression-prebn-correction-20260730 -->
## 2026-07-30 - LUT6 压缩与 pre-BN 来源的结论修正

### 对“6 输入压成 1 bit”的修正

上一条记录把 LUT6 的六变量高阶交互能力直接表述成比当前 LUT4 更强，容易误导。正确比较必须拆成两个维度：

1. **输出带宽/压缩率**：实数局部 LUT6 是 6 scalar input bits -> 1 bit，压缩率确实高；当前 pair-LUT 是两个复数，即 4 scalar input bits -> 2 bits，局部输出带宽更高、压缩更弱。从这个角度，不能说当前结构天然具有更严重的信息瓶颈。
2. **单个布尔函数的交互阶数**：一个 LUT6 的单输出表有 64 entries，可表达任意六变量布尔函数；当前每个 real/imag LUT4 只有 16 entries，各自表达任意四变量函数。LUT6 可以直接建模六个输入之间的高阶交互，而当前只能在固定“两复数一组”内建模四变量交互。
3. **层级输出并非最终 1 bit**：实数卷积 patch 被分成多个六输入组，每组各输出 1 bit，随后这些局部 bits 求和并进入 BN；不是整个 patch 被一个 LUT 压成 1 bit。当前也对多个 pair 的 real/imag bits 分别求和。因此两者都是“分组布尔函数 + 多组整数累加”，只是在组大小和每组输出数上不同。
4. **结论**：LUT6 有更强的单组非线性交互但更窄的局部输出；LUT4 pair 有更宽的局部输出但更低的单组交互阶数。仅凭 6->1 与 4->2 不能判断哪个总体表达瓶颈更大，更不能把当前 49% 平台直接归因于表达能力。后续应优先用 Phase2 truth-table 初始化测试结构上限，再区分表达瓶颈与随机 hard-LUT 优化失败。

### pre-BN 的历史来源

Git 历史显示：

- `complexPyTorch/complexBinaryResNet.py` 首次出现在 commit `228fb43 add binary`，从第一版起，单卷积 `BiRealComplexResidualBlock` 就是 `ComplexBN(pre) -> Sign -> BinaryComplexConv -> ComplexBN(post) -> residual add`。
- 原始同期的全精度 `ComplexResidualBlock` 是两卷积 pre-activation 结构：`BN1 -> ReLU -> Conv1 -> BN2 -> ReLU -> Conv2 -> residual add`。它确实在每个卷积前使用 BN，但第二个卷积后、残差相加前没有额外 post-BN。
- 二值改写把每个传统两卷积 block 展开成两个单卷积 Bi-Real blocks，同时既保留了每个单卷积前的 pre-BN，又新增 post-BN 来控制二值累加尺度。因此当前每个单卷积的双 BN 不是原始全精度 complex block 的严格等价展开，而是二值路线的主动设计。
- 当前 Phase1/2/3 共用这个 block；Phase1 只把 Sign/BinaryConv 换成 ReLU/ComplexConv，仍保留 pre+post BN。旧 Phase1/2 的高精度已经证明 pre-BN 并不会让普通 complex/BNN 必然训崩。
- 实数 LUT-BiReal block 没有 pre-BN，只有 `BinaryActivation -> LUT -> BatchNorm -> residual add`。因此 pre-BN 仍是 scratch pair-LUT 与实数参考的真实差异，可能通过持续重排 truth-table 地址和 complex covariance statistics 增加联合优化难度，但不能在没有消融实验时认定它是唯一原因。
- pre-BN 对 Bi-Real surrogate 也有潜在正面作用：它把 sign 前输入拉回约单位尺度，使 `2(1-|x|)` 的非零梯度窗口得到更多覆盖。直接删除 pre-BN 可能减少梯度，必须作为独立开关消融，不能无条件删除。

### 文件记录

本条仅通过 `scripts/append_experiment_log.py` 更新 `EXPERIMENT_LOG.md`；没有修改模型、训练器或启动命令。

<!-- experiment-entry:pre-post-complex-bn-hardware-20260730 -->
## 2026-07-30 - pre/post ComplexBN 的硬件代价与消融设计

### 当前实现确认

`BiRealComplexResidualBlock` 中：

- `bn_pre = ComplexBatchNorm2d(in_channels, eps=1e-4)`
- `bn_post = ComplexBatchNorm2d(out_channels, eps=1e-4)`
- projection 分支也使用 `ComplexBatchNorm2d`

这里的 ComplexBatchNorm2d 不是 real/imag 独立的 NaiveComplexBatchNorm2d，而是计算 real/imag 的 2x2 covariance whitening，并带有交叉仿射参数。推理阶段 running statistics 固定后，每个复数 channel 可合并为：

```
y_r = A00*x_r + A01*x_i + c0
y_i = A10*x_r + A11*x_i + c1
```

其中 A 同时包含 covariance whitening 和 affine matrix。也就是说，当前 pre/post BN 都需要 real/imag 交叉通路。

### 硬件友好性

**pre-BN 相对更容易折叠**。因为它后面立刻接 sign，实际不需要物化浮点 BN 输出，只需计算：

```
bit_r = 1[A00*x_r + A01*x_i + c0 >= 0]
bit_i = 1[A10*x_r + A11*x_i + c1 >= 0]
```

这可以实现成两个固定系数二维比较器。它仍比普通 threshold 昂贵，因为存在 real/imag cross term，但不需要保留归一化后的幅值。

若 pre-BN 改成 `NaiveComplexBatchNorm2d`，real/imag 完全独立，则 sign(BN(x)) 可严格折叠为每通道两个阈值比较（外加 scale 为负时的极性翻转），基本不需要乘法器。这是最硬件友好的形式。

**post-BN 更难折叠**。它的输出先与 residual 相加，幅值和尺度都会影响后续层，不能只保留 sign。当前结构实际需要：

```
out = A * [lut_count_r, lut_count_i] + c + residual
```

即使 A 和 c 都是推理期常数，也仍需在 LUT 整数累加结果上执行定点常系数乘加；cross term 还会让 real LUT count 参与 imag 输出、imag count 参与 real 输出。可以用 DSP、shift-add 或加权 adder tree 实现，但不能像 pre-BN 那样简单退化为阈值。

post-BN 也不能直接与下一层 pre-BN 完全相消，因为二者之间夹着 residual add；代数组合后仍需要分别缩放 LUT branch 和 identity branch。

### 训练角度

硬件更友好不等于训练中更不重要：

- pre-BN 将 sign 输入拉回单位尺度，有助于 Bi-Real surrogate `2(1-|x|)` 保持非零，但也会随 running/affine 参数变化持续改变 LUT 地址分布。
- post-BN 负责归一化大量局部 0/1 LUT outputs 的整数累加，并校准 residual branch 尺度。实数 LUT-BiReal 保留的正是 post-BN，因此完全删除 post-BN 的训练风险高于删除 pre-BN。
- 当前日志不能判断是哪一个 BN 导致 49% 平台；旧 Phase1/2 同时使用二者仍能训练，说明二者都不是普遍错误，但随机 hard-LUT scratch 下的联合动态可能不同。

### 推荐消融顺序

不要第一步同时删除两个 BN。建议固定优化器、seed、LUT 初始化和其他结构，按以下顺序：

1. baseline：covariance pre + covariance post（当前）。
2. no-pre：无 pre-BN + covariance post。只回答额外 pre-BN 是否妨碍 hard-LUT scratch，并更接近实数 block。
3. naive-pre：naive/diagonal pre + covariance post。若它优于 baseline，说明问题主要来自 pre-BN 的 real/imag covariance cross coupling，而不是 pre-normalization 本身。
4. covariance pre + naive/diagonal post。测试 post-BN 的 cross coupling，同时保留 LUT count 的尺度归一化。
5. no-post 只作为最后的高风险实验；若测试，应同时加入简单可学习/定点友好的 branch scale，否则 LUT count 与 residual 尺度失配会混淆结论。

从硬件与训练的折中看，最有希望的最终形式是 **naive pre-BN（折叠成阈值）+ naive post-BN（两个独立定点 scale/offset）**；是否需要 pre-BN 本身由 no-pre 对照决定。

### 文件记录

本条只通过 `scripts/append_experiment_log.py` 更新 `EXPERIMENT_LOG.md`，没有修改模型代码。

<!-- experiment-entry:bn-mode-ablation-20260730 -->
## 2026-07-30 - Configurable residual pre/post BatchNorm ablation

### 目标与实验假设

当前 complex Bi-Real residual block 同时包含 convolution 前的 pre-BN 和 convolution 后的 post-BN；二者默认均为 real/imag covariance whitening。为区分训练不动究竟来自额外 pre-normalization、real/imag cross coupling，还是 post-BN 本身，本次将 pre/post normalization 解耦为独立开关。

每个位置支持三种模式：

- covariance：原有 ComplexBatchNorm2d，执行 real/imag 2x2 covariance whitening 和交叉 affine。
- naive：NaiveComplexBatchNorm2d，real/imag 分别执行普通 BatchNorm，不发生交叉耦合。
- none：nn.Identity，完全移除该位置的归一化。

默认仍为 pre=covariance、post=covariance，因此不带新参数的旧命令、当前 GPU0 基线以及旧 checkpoint 架构均保持原行为。post 模式同时作用于 main path 的 bn_post 和 projection shortcut 的 BN，确保每个 post-BN 消融配置定义完整。stem 的 bn1 暂时保持 covariance，不混入本轮 residual-block 消融。

### 修改文件

- complexPyTorch/complexBinaryResNet.py：新增统一 BN factory；BiRealComplexResidualBlock 和 BinaryComplexResNet 接收 pre_bn_mode/post_bn_mode；所有 stage block 透传开关；projection BN 跟随 post 模式。
- training.py：新增 --pre-bn-mode 与 --post-bn-mode CLI，choices 为 covariance/naive/none；build_model 透传到网络。
- run_phase3.sh：新增 PRE_BN_MODE/POST_BN_MODE 环境变量、CLI 透传和启动时配置打印。
- run_phase3_real_compatible.sh：显式保留 covariance/covariance 默认，同时允许外部环境变量覆盖。
- tests/test_phase12_route.py：覆盖默认值、全部九种结构组合、projection 路由、naive/none 前后向以及非法模式拒绝。
- README.md：加入用户入口、模式说明和示例。
- CURRENT_TECHNICAL_ROUTE.md：记录开关语义、projection 约束和 stem 不参与本轮切换。
- EXPERIMENT_LOG.md：由 scripts/append_experiment_log.py 追加本条结构化记录。

### 验证

- python -m py_compile complexPyTorch/complexBinaryResNet.py training.py tests/test_phase12_route.py：通过。
- bash -n run_phase3.sh run_phase3_real_compatible.sh：通过。
- git diff --check：通过。
- CUDA_VISIBLE_DEVICES=8 conda run -n lut_net python -m unittest discover -s tests -v：25 tests 全部通过，包括 binary/floating LUT CUDA forward/backward equality。
- 默认 start_filter=11、num_blocks=3 的 Phase 3 参数量：cov/cov 2,031,663；none/cov 2,029,518；naive/cov 2,031,234；cov/naive 2,031,124；naive/naive 2,030,695；cov/none 2,028,968。

### 并行实验矩阵

GPU0 已有的 phase3_pair_lut_real_aligned_binary_clean 是 covariance/covariance 基线，继续保留。

- GPU1：none/covariance。回答额外 pre-BN 是否阻碍 hard pair-LUT scratch。
- GPU2：naive/covariance。与 GPU1 联合判断 pre-BN 的作用来自归一化本身还是 covariance cross coupling。
- GPU3：covariance/naive。单独检验 post-BN cross coupling，同时保留累加尺度归一化。
- GPU4：naive/naive。硬件友好的完整候选，pre 可折叠为独立阈值，post 为独立定点 scale/offset。
- covariance/none 风险最高：post-BN 负责 LUT 累加和 residual branch 的尺度校准，因此暂不列入第一批，除非前四组提示 post-BN 本身有害。

所有实验必须使用唯一 WORKDIR，防止 checkpoint 和日志覆盖；其余 real-compatible 参数与 GPU0 基线完全相同，使用相同默认 seed，便于单变量比较。

<!-- experiment-entry:bn-ablation-results-20260730 -->
## 2026-07-30 - Residual BN ablation results and remaining real-LUT differences

### 已完成的 BN 消融结果

使用 scripts/analyze_training_run.py 统一解析完成实验。四组均采用 real-compatible hard-forward identity-STE、Adam、普通参数与 LUT LR=0.01 线性衰减、batch size 256、50k train、相同 seed 和独立 workdir。

| pre-BN | post-BN | best test | best epoch | final test | best train | final LUT sign diff |
|---|---:|---:|---:|---:|---:|---:|
| none | covariance | 51.33% | 205 | 50.12% | 41.78% | 23.25% |
| naive | covariance | 49.77% | 219 | 48.10% | 39.46% | 22.85% |
| covariance | naive | 51.11% | 251 | 50.87% | 41.13% | 22.57% |
| naive | naive | 48.85% | 239 | 47.36% | 39.04% | 22.64% |

旧 covariance/covariance clean run 在 epoch 130 被提前停止，已达到 best 50.03%，不能作为完整 256-epoch final，但平台与四组一致。

结论：

1. 删除 pre-BN、将 covariance whitening 改为 real/imag 独立 naive BN，均只能产生约 1-2 个百分点波动，无法解释与实数 LUT 网络约 30-35 个百分点的差距。
2. 四组最终都有约 22.5%-23.2% LUT entry sign flip，说明 LUT 参数并非不更新。问题是大量离散变化只形成约 39%-42% augmented train accuracy，属于表示/优化方向不足，而不是 LUT 冻结。
3. covariance/none 仍作为最后一个高风险 BN 对照，命令为：
   GPU_ID=0 PRE_BN_MODE=covariance POST_BN_MODE=none WORKDIR=runs/phase3_bn_covpre_nopost ./run_phase3_real_compatible.sh

### 与实数 LUT-BiReal 已确认对齐的项目

- 数据：CIFAR-10 50k train、RandomCrop、HorizontalFlip、CIFAR10 AutoAugment、相同 mean/std、label smoothing 0.1。
- optimizer：Adam，无 weight decay，无 gradient clipping。
- schedule：初始 LR 0.01，256 epoch linear decay；LUT 与普通参数使用同一 LR。
- LUT logit：50/50 bimodal Normal(-1,0.2) / Normal(+1,0.1)。
- LUT forward：从 epoch 1 使用 hard 0/1 entry，logit 使用 identity STE。
- activation 总梯度：实数 BinaryActivation 的 0/1 surrogate 导数为 1-|x|；复数路径先以 +/-1 输出产生 2(1-|x|)，进入 pair-LUT probability 编码时乘 1/2，合成后同样为 1-|x|。
- binary CUDA 与 floating multilinear backend 已有 exact forward/backward corner regression test，因此当前 binary packing 不是首要嫌疑。

### 仍未对齐且优先级更高的差异

1. LUT fan-in 与局部函数阶数。实数网络每个 neuron 使用 LUT6：6 个 scalar activation bits 到 1 bit，单表 64 entries。当前 pair-LUT 使用两个复数 activation，也就是 4 bits，并分别输出 real/imag bit：两张 LUT4 合计只有 32 entries。当前每个输出分量只能学习 4-bit interaction，不能复现实数 LUT6 的 6-bit interaction。
2. Boolean 参数容量。实数 BiRealNet20 的 LUT entries 约 2,850,816；当前 start_filter=11 pair-LUT 为 2,022,592，只有约 71%。更关键的是，当前每个物理双输出 LUT 只使用 32 个 INIT 自由度，而 LUT6 单输出使用 64 个。
3. 物理利用率。按当前 stage 宽度估算，pair-LUT 需要约 63,206 个 4-input/2-output physical neurons；实数 16-channel LUT6 网络约 44,544 个。当前方案可能消耗更多 physical LUT，却因为只使用 4 个输入而获得更低的局部阶数。
4. 网络前端。实数版是 real Conv stem；当前先用 LearnImagBlock 从 RGB 学习 imaginary，再走 ComplexConv + covariance stem BN。
5. projection。实数下采样 shortcut 为 AvgPool -> 1x1 Conv -> BN；当前是 stride-2 ComplexConv -> selected post-BN。
6. 宽度。实数网络 start width=16 real channels；当前为 11 complex channels，不能只按 channel 数直接等价比较。
7. 随机训练能力尚无控制组。现有 canonical Phase 2 的 82.21% 来自 Phase 1 checkpoint，不是同配方 scratch。必须先判断 complex BNN 本身能否用当前 scratch recipe 训练。

### 下一步控制实验

与 no-post 并行跑 Phase 2 scratch，保持 no-pre/cov-post 和实数训练配方：

GPU_ID=1 TRAIN_FROM_SCRATCH=1 NUM_EPOCHS=256 BATCH_SIZE=256 LR=0.01 SCHEDULE=linear WEIGHT_DECAY=0 WORKDIR=runs/phase2_scratch_real_recipe_nopre_covpost ./run_phase2.sh --optimizer adam --clipnorm 0 --clipval 0 --augmentation real_lut --label-smoothing 0.1 --no-validation --pre-bn-mode none --post-bn-mode covariance

判据：

- 若 Phase 2 scratch 也约 50%，先解决 complex backbone scratch optimization，不能继续归因于 pair-LUT。
- 若 Phase 2 scratch 达到约 80%，则主要瓶颈就是 LUT4 pair grouping。下一版优先考虑将全部 real/imag scalar bits flatten 后每 5 bit 一组，用一块 LUT6_2 实现两个独立 LUT5 outputs。这样每个物理单元拥有 32+32=64 entries，与实数 LUT6 的 64 个布尔自由度相当，同时 physical neuron 数比当前 4-bit grouping 更少；输出仍可解释为一个复数的 real/imag 两个 bit。

<!-- experiment-entry:phase2-scratch-control-20260730 -->
## 2026-07-30 - Phase 2 scratch control reaches 80.96 percent

### 完成结果

使用 scripts/analyze_training_run.py 解析 runs/phase2_scratch_real_recipe_nopre_covpost：

- 状态：complete，256 epochs。
- best test accuracy：80.96%，epoch 255。
- final test accuracy：80.18%。
- best epoch augmented train accuracy：75.26%。
- 配置：Phase 2、train from scratch、pre-BN=none、post-BN=covariance、Adam、LR=0.01 linear decay、batch size 256、weight decay=0、无 gradient clipping、real_lut augmentation、label smoothing 0.1、50k train/no-validation。

### 归因更新

该控制组与随机 hard pair-LUT 的最佳 51.33% 相差 29.63 个百分点，而 backbone、数据、optimizer、schedule、activation surrogate、stem/projection 和 BN 模式保持一致。由此可以排除以下主因：

- complex backbone 无法从头训练；
- no-pre/cov-post 结构本身错误；
- Adam 0.01 linear 配方不适用于复数网络；
- 数据增强或 label smoothing 导致约 50% 平台。

问题已经收敛到 BinaryComplexConv2d 被随机 4-input/2-output PairLUTNeuronConv2d 替换后的表示与优化：

1. 随机 hard LUT sign 在前几十个 epoch 快速大规模翻转，缺少二值卷积提供的连续 latent spatial-weight 参数化和结构先验。
2. 每个输出分量只有 4-bit interaction；实数成功路线是 6-bit interaction。
3. 当前每个物理双输出单元只有两张 LUT4，共 32 个 entry bits；实数 LUT6 为 64。
4. 当前 pair grouping 的 truth tables 完全独立，参数空间维度高但缺少相邻 spatial position 或 complex multiplication 的共享约束，因此 identity STE 的离散搜索难度大。

### 下一步实验

先使用本次最佳 Phase 2 checkpoint 初始化 Phase 3 pair-LUT，而不是随机 bimodal：

GPU_ID=2 TRAIN_FROM_SCRATCH=0 CHECKPOINT=runs/phase2_scratch_real_recipe_nopre_covpost/chkpts/Bestmodel_phase2.pt PRE_BN_MODE=none POST_BN_MODE=covariance WORKDIR=runs/phase3_pair_from_phase2_scratch80_hard ./run_phase3_real_compatible.sh

该实验保持 hard-forward identity STE，但初始真值表来自已达到 80.96% 的二值卷积。它回答两个问题：

- pairwise local threshold 编译本身造成多少即时 accuracy 损失；
- 有结构先验的 LUT 初始化能否避开随机 hard LUT 的 50% 平台。

若初始化后立即显著高于随机版本并能继续恢复，短期路线采用 Phase2 -> Phase3。若仍迅速降到约 50%，则 4-bit pairwise local function 本身是主要表示瓶颈，应转向 flatten scalar bits 后的 5-input/2-output LUT6_2 neuron。

<!-- experiment-entry:phase2-pairlut-non-equivalence-20260730 -->
## 2026-07-30 - Correction: Phase 2 to pair-LUT is not an equivalent conversion

### 对上一条归因的修正

Phase 2 checkpoint 可以为 Phase 3 提供有结构的 truth-table 初始化，但 Phase 2 与当前 pair-LUT Phase 3 的 forward 并不等价，不能将该转换描述为保持 Phase 2 表达。

Phase 2 BinaryComplexConv2d 先在完整 kernel 范围内累加所有二值复乘：

Y_r = alpha * sum_j (x_rj*w_rj - x_ij*w_ij)
Y_i = alpha * sum_j (x_rj*w_ij + x_ij*w_rj)

每个复乘的实部/虚部分量属于 {-2,0,2}，但完整卷积会保留所有位置累加后的多比特整数值，随后才进入 post-BN。

当前 PairLUTNeuronConv2d 先将位置按两个复数 activation 分组。每组先计算两个复乘之和：

s_p,r / s_p,i 属于 {-4,-2,0,2,4}

随后初始化 truth table 时立即执行 >=0，分别压成 real/imag 1 bit。hard forward 中每个 pair 对每个输出分量只贡献 +/-2*alpha，最终再累加所有 pair：

Y'_r = 4*alpha*(sum_p bit_p,r - P/2)
Y'_i = 4*alpha*(sum_p bit_p,i - P/2)

因此局部映射发生不可逆合并：

- -4 与 -2 都映射到负 bit；
- 0、2、4 都映射到正 bit；
- tie=0 固定归入正侧；
- 每组贡献的原始 magnitude 和 zero state 在全局累加前已经丢失。

post-BN 只能重新校准最终 count 的均值和尺度，不能恢复每个 pair 被丢掉的 magnitude/zero 信息。Phase 2 -> Phase 3 checkpoint conversion 的含义只是使用 Phase 2 weight signs 构造一个合理的 Boolean 起点，不是无损编译。

### 正确的判别实验

直接运行 Phase 2 checkpoint 初始化的 Phase 3 仍有价值，但它同时混合了两个因素：

1. random hard LUT optimization 是否困难；
2. pairwise local 2-bit truncation 是否具有足够表示能力。

要将二者拆开，需要新增一个固定解析算子的控制分支：

- 保留可训练 BinaryComplexConv2d latent weights；
- forward 对 activation/weight 做与 Phase 2 相同的二值化；
- 每两个复数位置计算两次复乘并求和；
- 对 pair sum 的 real/imag 分别执行 >=0 hard bit，并给 comparator 配置 STE；
- 再像 PairLUTNeuronConv2d 一样对所有 pair bit count 做中心化累加；
- 不使用任何可学习 LUT entries。

该 analytic pair-comparator forward 与由当前 weight signs 枚举得到的 hard pair-LUT truth tables逐状态严格一致，同时优化变量仍是稳定的 latent spatial weights。

判据：

- analytic pair-comparator 从头也只能约 50%：主要瓶颈是局部 2-bit 截断/4-bit fan-in 的表示损失；
- analytic 分支能到约 80%，随机 pair-LUT 约 50%：主要瓶颈是自由 truth-table 的离散优化；
- analytic 分支高，但转换到同权重 hard LUT 后不一致：说明 LUT grouping、bit ordering、padding 或 scaling 实现存在 bug。

因此下一步最干净的工作不是继续调 BN，而是实现这一 analytic pair-comparator 控制开关，并增加逐状态及整层 forward equality test。

<!-- experiment-entry:analytic-pair-control-20260730 -->
## 2026-07-30 - Analytic pair-comparator control implementation

### 目标

新增一个不学习自由 truth-table 的 Phase 3 控制分支，单独测试“每两个二值复数乘法求和后，real/imag 各截断为 1 bit，再累加”的表示能力。默认 Phase 3 LUT 路径保持不变。

### 数学与梯度

AnalyticPairComparatorConv2d 保留与 BinaryComplexConv2d 相同的 latent conv_r/conv_i weights。每次 forward：

1. 对 latent weights 和 complex activation 的 real/imag 分别 hard sign。
2. 按 [input channel, ky, kx] flatten 后每两个 complex positions 分组。
3. 每组计算两个 complex products 的和。
4. real/imag pair sum 分别执行 >=0，得到 +/-1。
5. 每个 pair 乘 2*alpha 后跨 pair 累加。

hard forward 通过由当前 weight signs 即时生成的 frozen pair truth tables执行，复用 packed binary backend，只是计算实现，不包含 trainable LUT entries。它与部署时的 PairLUTNeuronConv2d hard forward逐状态一致，包括 padding 的 fixed-low bit 和 odd-tail dummy。

backward 使用解析 identity-STE comparator。因为每个 pair output 为 2*alpha*STE_sign(pair_sum)，且 d STE_sign / d pair_sum = 1，跨 pair 求和后的代理梯度等价于两倍的 binary complex convolution。实现使用自定义 _HardForwardProxyBackward：forward 原样返回 hard LUT output，backward 只将梯度传给 latent-weight analytic proxy。这样避免普通 detach 加减表达式产生约 2.7e-7 forward rounding error。

### 新增开关与命令

- --phase3-mode lut：原有默认路径。
- --phase3-mode analytic_pair：新控制路径。
- run_phase3_pair_analytic.sh：默认 train from scratch、256 epochs、batch 256、Adam、LR 0.01 linear、weight decay/clipping 0、real_lut augmentation、label smoothing 0.1、50k train、pre-BN none、post-BN covariance、packed binary hard forward。

运行命令：

GPU_ID=2 WORKDIR=runs/phase3_pair_analytic_scratch ./run_phase3_pair_analytic.sh

### 修改文件

- complexPyTorch/complexLayers.py：PairLUTNeuronConv2d 增加 trainable_entries；新增 hard-forward/proxy-backward autograd bridge 和 AnalyticPairComparatorConv2d。
- complexPyTorch/complexBinaryResNet.py：Phase 3 block 根据 phase3_mode 选择自由 PairLUTNeuronConv2d 或 AnalyticPairComparatorConv2d；默认 lut 不变。
- training.py：新增 CLI 并透传；analytic 模式跳过 LUT 初始化、退火校验、LUT LR 日志、sign diff 和 hard-table checkpoint 字段；普通 checkpoint/best 保存仍工作。
- run_phase3.sh：透传 PHASE3_MODE，并按模式打印准确的初始化/训练语义。
- run_phase3_pair_analytic.sh：新增独立的一键控制实验入口。
- tests/test_pair_lut_phase3.py：新增含 padding+odd tail 的 exact hard-forward equality、input/latent-weight gradient、无 trainable LUT entries 和训练路由测试。
- README.md：新增用户命令与简要语义。
- CURRENT_TECHNICAL_ROUTE.md：记录该控制分支的数学定义、用途与判据。
- EXPERIMENT_LOG.md：由 scripts/append_experiment_log.py 追加本条。

### 验证

- 定向 3 tests：全部通过。
- CUDA_VISIBLE_DEVICES=8 conda run -n lut_net python -m unittest discover -s tests -v：28 tests 全部通过。
- 默认 start_filter=11、num_blocks=3、batch=2 的 GPU packed-binary analytic model forward/backward：output shape (2,10)，96 个 trainable parameters tensors 获得 gradient，总 trainable scalar parameters 259,574。
- py_compile、bash -n、git diff --check：通过。
- PYTHON_BIN=/bin/echo dry-run：确认完整 invocation 包含 phase3_mode=analytic_pair、binary kernel、Adam 0.01 linear、no-pre/cov-post、no-validation 和 train-from-scratch。

### 结果判据

- 若 analytic_pair 仍约 50%：局部 two-product -> two-bit 截断或 4-bit fan-in 是主要表示瓶颈。
- 若 analytic_pair 接近 Phase 2 scratch 的 80.96%，而自由 pair-LUT 仍约 50%：直接优化独立 truth tables 是主要瓶颈。
- 若 analytic hard forward 与单独转换的 PairLUT 出现偏差：属于 grouping/bit-order/padding/scale bug；现有 exact equality regression 已覆盖该风险。

<!-- experiment-entry:analytic-pair-control-verification-20260730 -->
## 2026-07-30 - Analytic pair control verification supplement

### 审计脚本同步

新增 run_phase3_pair_analytic.sh 后，scripts/audit_active_route.py 的旧启动器允许列表首先按预期失败。已更新可复用审计脚本：

- 将 run_phase3_pair_analytic.sh 纳入 Phase 1/2/3 活动路线；
- 验证 analytic_pair 模型实际包含 AnalyticPairComparatorConv2d；
- 验证不存在 trainable pair_lut.lut_r/lut_i；
- 验证 latent conv_r/conv_i weights 存在。

在 lut_net 环境重新运行审计后全部通过。scripts/audit_active_route.py 是本次新增修改文件，补充到上一条文件记录。

### 正式 batch size 显存检查

在空闲 GPU8 上使用默认完整模型 start_filter=11、num_blocks=3、phase3_mode=analytic_pair、binary kernel、no-pre/cov-post 和随机 batch size 256，完成一次 forward/backward：

- output shape：(256, 10)
- peak torch CUDA allocated memory：1.701 GiB
- 无 OOM 或 kernel error

因此专用 launcher 默认 batch size 256 可直接作为第一组正式实验。

<!-- experiment-entry:real-vs-complex-hard-lut-audit-20260730 -->
## 2026-07-30 - Real-vs-complex hard-LUT function audit

### 问题

实数域 LUT-BiReal 使用局部硬 0/1 LUT 后可以从头训练，而当前复数 LUT-as-neuron 路线在硬局部输出下长期停在约 50%。本次不再只比较模块名称，而是逐项核对 forward、backward、局部函数族、残差拓扑和训练结果。

### 对齐的函数

1. **激活前向与代理梯度**
   - 实数域 `BinaryActivation` 前向为 `x > 0` 的 0/1 bit，反向导数在 `|x| < 1` 内为 `1-|x|`。
   - 复数自由 pair-LUT 路径先用 `BinaryComplexActivation` 产生 -1/+1，导数为 `2(1-|x|)`，再在 `PairLUTNeuronConv2d` 中映射 `(x+1)/2`。合成导数同样是 `1-|x|`。
   - 因而自由 pair-LUT 的激活代理梯度与实数参考一致。

2. **硬表项与 logit 梯度**
   - 实数域和复数 `real_compatible` 都用 `logit >= 0` 得到硬 0/1 entry，并对 logit 使用 identity STE。
   - Adam、初始 LR 0.01、线性衰减、无 LUT weight decay、bimodal `N(-1,0.2)` / `N(+1,0.1)` 初始化均已对齐。

3. **LUT 输入与 entry backward**
   - 两个 binary CUDA kernel 都给命中的 entry 累积上游梯度。
   - 输入 bit 梯度都是相邻表项有限差分 `table(bit=1)-table(bit=0)`。
   - 当前测试覆盖了 complex binary kernel 和浮点多线性参考的 forward/backward 一致性。因此暂未发现 CUDA backward 语义错误。

4. **输出求和与 BN**
   - 实数域先求和 0/1 LUT 输出，再由 post-BN 去均值和缩放。
   - 复数域先做 `4*(count-P/2)` 再进入 post-BN。只要 post-BN 存在，常数平移和固定缩放在训练态原则上被 BN 消除。
   - covariance/naive、pre-BN on/off 四组消融都约为 49%-51%，也说明 BN 类型不是主因。

### 真正不等价的函数

1. **局部函数容量不是由压缩比决定**
   - 一个任意 LUT6->1 有 64 个独立 INIT bit，可表示 `2^64` 个 Boolean 函数，并产生六阶输入交互。
   - 当前一个物理 pair neuron 是共享四个输入的两个 LUT4->1，一共只有 32 个独立 INIT bit，可表示 `2^32` 个双输出映射；每个输出只看到四个 bit。
   - 所以 4->2 看似比 6->1 压缩轻，但局部函数族反而小了平方级。当前实现还留下了 LUT6_2 的输入/INIT 容量未使用。

2. **解析复数 comparator 不是实数自由 LUT**
   - 两个二值复数乘积的单个分量之和只可能为 `{-4,-2,0,2,4}`，概率分别为 `{1/16,4/16,6/16,4/16,1/16}`。
   - 精确零占 `37.5%`；使用 `>=0` 后正 bit 占 `11/16=68.75%`。
   - 枚举 checkpoint 得到 hard ones `0.6876`、Boolean sensitivity `0.3748`，与理论完全一致。
   - 解析层的一个权重 sign 控制一批 truth-table entries；它跨零会同时改变大量地址。实数自由 LUT 则每个 entry 独立跨零。
   - 当前解析层反向使用连续 binary complex convolution 作为 dense proxy；实数 LUT 的输入反向是硬表有限差分。两者并不相同。

3. **自由实/虚 LUT 破坏了原复数结构**
   - Phase 2 的实部和虚部由同一组 `w_r,w_i` 通过复数乘法共同约束，输出与后续 complex BN、complex projection、complex residual 的语义一致。
   - 随机 pair-LUT 的实表和虚表完全独立，但网络其余位置仍使用受限的复数线性结构。模型必须同时学习分类函数和重新协调两个输出流。
   - 最佳硬模型的实/虚表 Hamming 距离为 `0.5014`，说明训练后仍接近独立随机函数，没有形成明显的成对结构。

4. **残差投影仍未完全复制实数网络**
   - 实数参考下采样为 AvgPool -> unrestricted real 1x1 Conv -> BN。
   - 当前复数网络为 stride-2 complex 1x1 Conv -> complex/naive BN。
   - Phase 2 对照能达到 80.96%，所以这不是硬 LUT 失败的单一原因，但在 arbitrary paired LUT 输出已经不再遵守复数乘法时，这个结构约束可能成为重要放大器。

### 关键结果

- 实数参考 `log_lr0p01.txt`: best 87.50% at epoch 132；`log_2x.txt`: best/final 83.47% at epoch 256。
- 对齐实数训练 recipe 的复数 Phase 2 scratch: best test 80.96%。
- 硬随机 pair-LUT，no-pre/cov-post: best test 51.33%，final 50.12%，final train 41.78%。
- 硬解析 pair-comparator scratch: best test 50.22%，final 49.70%，final train 40.59%。
- 从 Phase 2 初始化的 soft pair-LUT 在 44 epoch 内 best 71.77%，但 hard sign diff 始终为 0；其提升来自连续 soft table，而不是学出了新硬函数。

### Checkpoint 结构审计

新增 `scripts/analyze_pair_lut_checkpoint.py` 后分析 no-pre/cov-post 最佳模型：

- 126,412 个标量 Boolean 函数，2,022,592 个 entries。
- sign flip `470202/2022592 = 23.25%`，说明 entry 确实在更新。
- ones `0.5009`，常量函数约 0，平均 sensitivity `0.4972`。
- 实/虚 Hamming `0.5014`。
- 结论：表没有冻结或常量化，但整体 Boolean 结构仍与随机初始化极其相似，更新没有形成协调的局部特征族。

新增 `scripts/analyze_pair_lut_states.py`，在最佳模型和 512 张测试图上统计地址：

- 平均 entropy `3.3315/4 bit`，等效 `10.73/16` 个状态。
- 平均观察到 `98.86%` 的地址。
- 第一层相关性较强，但后层大多数 pair 的 16 个地址都被访问。
- 结论：训练失败不能主要归因于大量 LUT entry 从未被访问。

### 当前结论

“复数域不能训练硬截断”并不准确。当前失败来自两个不同机制：

- 解析路线保留复数语义，但局部多数比较器有 37.5% tie、强信息压缩、粗粒度权重 sign 跳变和不匹配的 dense proxy。
- 自由 LUT 路线使用了正确的实数式 STE，却把 LUT6->1 换成容量更小的共享输入 LUT4x2，并让独立实/虚 Boolean 函数继续穿过受限复数骨架。它不是实数网络的逐函数复刻。

### 下一步判别顺序

1. **精确 scalarized LUT6 control**：把 real/imag 拼成 `2C` 个普通标量 channel，完全复制实数 LUT6->1、real BN、real projection 和 residual；输出 `2O` 后再成对解释。若该控制恢复到 80% 左右，根因就是当前 LUT4x2/复数骨架，而非硬 LUT 训练本身。
2. **硬件候选 LUT5_2**：每个物理 LUT6_2 共享 5 个 activation bit，产生两个独立 5-input 输出。它有 64 个 INIT bit，与实数 LUT6->1 的参数容量相同，同时比当前 LUT4x2 使用更少的物理 cell group。
3. 在完成上述 topology control 前，不再优先扫 LR、tau 或 BN；现有证据已经表明这些不是决定性变量。

### 本次文件改动

- `scripts/analyze_pair_lut_checkpoint.py`：可复用的 hard-table 结构、sign diff、Boolean sensitivity、实虚 Hamming 和 logit margin 分析。
- `scripts/analyze_pair_lut_states.py`：可复用的真实数据 LUT 地址 occupancy、entropy 和 bit balance 分析。

<!-- experiment-entry:standard-complex-bireal-learnimag-20260730 -->
## 2026-07-30 - Standard complex Bi-Real baseline with CIFAR LearnImagBlock

### Goal

Establish a clean Phase 2 control that reproduces the standard CIFAR Bi-Real residual trunk in the complex domain before returning to the LUT comparison. Per the clarified requirement, CIFAR's real-valued image input must still pass through `LearnImagBlock`; only the trunk after that complex frontend is aligned to Bi-Real.

### Design

- Added `--bireal-topology {legacy,standard}`, defaulting to `legacy`. Existing commands, checkpoints, and Phase 3 experiments therefore retain their previous topology unless the new mode is requested explicitly.
- The `standard` block is `BinaryComplexActivation -> BinaryComplexConv2d -> complex post-BN -> residual add`. It has no pre-BN.
- Stage 2 has no projection when shape is unchanged. Stage 3/4 shortcuts use `ComplexAvgPool2d(2, stride=2) -> full-precision ComplexConv2d(1x1, stride=1) -> complex BN`, matching the real CIFAR Bi-Real topology.
- `num_blocks=3` still expands to six one-convolution Bi-Real blocks per stage, matching the reference ResNet-20 layout.
- The existing signed activation already matches the standard Bi-Real forward and proxy backward: `sign(x)` in forward and `2 * max(1 - abs(x), 0)` in backward.
- Added the exact Bi-Real weight proxy for standard mode: forward uses detached `alpha * sign(w)`; backward passes through `clamp(w, -1, 1)` directly. This removes the extra `alpha` multiplier that the legacy STE applies to weight gradients.
- Intentional complex-domain differences remain: `LearnImagBlock`, complex activation/weights, complex convolution arithmetic, complex BN, and concatenated real/imaginary features at the classifier. The dedicated launcher defaults to covariance complex BN, but `POST_BN_MODE` remains configurable.

### Training Configuration

Added `run_phase2_complex_bireal.sh`. Its defaults reproduce the real reference training setup where applicable:

- Phase 2 from scratch; CIFAR `LearnImagBlock` enabled.
- 256 epochs, batch size 128, start filters 16, three logical blocks (six Bi-Real blocks per stage).
- Adam, initial LR `0.001`, weight decay 0, no gradient norm/value clipping.
- New `bireal_reference` schedule with 0.1 decays at epochs 90, 140, 180, and 220.
- CIFAR AutoAugment-compatible `real_lut` augmentation, label smoothing 0.1, and train/test-only evaluation via `--no-validation`.
- Run command: `GPU_ID=1 WORKDIR=runs/phase2_complex_bireal_gpu1 ./run_phase2_complex_bireal.sh`.

### Modified Files

- `complexPyTorch/complexFunctions.py`: added the selectable `bireal` binary-weight forward/backward proxy while retaining `scaled_ste` as the default.
- `complexPyTorch/complexLayers.py`: wired `weight_proxy_mode` into binary complex convolution and the analytic comparator without changing old defaults.
- `complexPyTorch/complexBinaryResNet.py`: added the selectable standard topology, preserved `LearnImagBlock`, removed pre-BN only in standard mode, corrected Stage 2 projection, and implemented AvgPool-first downsample shortcuts.
- `training.py`: exposed `--bireal-topology`, passed it into the model, and added the exact reference LR milestones.
- `run_phase2_complex_bireal.sh`: added the executable standard-baseline launcher.
- `tests/test_phase12_route.py`: added topology, frontend, shortcut, state-key, weight-gradient, schedule, and forward/backward coverage.
- `scripts/audit_active_route.py`: added the launcher to the active-route inventory and structural checks for the standard complex Bi-Real route.
- `EXPERIMENT_LOG.md`: recorded this implementation, rationale, command, and verification.

### Verification

- `bash -n run_phase2_complex_bireal.sh`: passed.
- Python compilation of all changed Python files: passed.
- `conda run -n lut_net python -m unittest tests.test_phase12_route`: 14 tests passed.
- `conda run -n lut_net python -m unittest tests.test_pair_lut_phase3`: 16 tests passed, confirming the legacy Phase 3 path remains intact.
- `conda run -n lut_net python scripts/audit_active_route.py`: passed all route and topology checks.
- `conda run -n lut_net ./run_phase2_complex_bireal.sh --help`: launcher reached the training CLI successfully with the intended defaults.
- No long training run was started by this change; accuracy must be evaluated from the new workdir before using this topology as the Phase 3 baseline.

<!-- experiment-entry:archive-before-clean-four-lutconv-20260801 -->
## 2026-08-01 - Archive boundary before clean four-LUTConv Bi-Real route

### Confirmed Baseline

- `runs/phase2_complex_bireal_gpu1` completed 256 epochs.
- Best test accuracy is `85.05%` at epoch 242; final test accuracy is `84.65%`.
- This validates the standard complex Bi-Real trunk with the CIFAR `LearnImagBlock`, no residual pre-BN, and reference Bi-Real optimizer/schedule.
- The result is now the only Phase 2 baseline for the next route.

### Current Source Snapshot

The working tree contains two groups of relevant source changes that must be archived before cleanup:

- Standard complex Bi-Real alignment in `complexFunctions.py`, `complexLayers.py`, `complexBinaryResNet.py`, `training.py`, the Phase 2 launcher, tests, and route audit.
- New differentiable LUTConv CUDA work in `complexPyTorch/lut_backend.py`, `lut_cuda/setup.py`, the tracked LUT CUDA backends, and the new BAFW/BFFB CUDA sources.

The current source also still contains the previous pair-LUT route:

- `PairLUTNeuronConv2d` and `AnalyticPairComparatorConv2d`.
- Pair-specific Phase 2-to-3 conversion, random initialization, annealing, sign-diff capture, checkpoint table capture, scripts, launchers, tests, and audit assertions.
- `BinaryLUTComplexConv2d` exists as an unwired development class. At this archive point, `complexBinaryResNet.py` still instantiates `PairLUTNeuronConv2d` for Phase 3.

### New Route Definition

After this snapshot is committed and pushed to an archive branch, a new local-only route will be cleaned to the following contract:

- Phase 1: full-precision complex network.
- Phase 2: standard complex Bi-Real network.
- Phase 3: exactly the same Bi-Real topology, normalization, residual, projection, stem, and classifier as Phase 2; only each `BinaryComplexConv2d` is replaced by a four-branch LUT complex convolution.
- The four real-valued branches are `x_r -> y_r`, `x_i -> y_r`, `x_i -> y_i`, and `x_r -> y_i`, combined as the standard complex expression `y_r = rr - ii`, `y_i = ir + ri`.
- Pair-LUT code and its command-line surface will be removed from the clean version.
- The cleaned route will remain local and unpushed until its behavior is verified.

### Git Scope

The archive commit will include source, CUDA source, scripts, tests, launcher, and this experiment record. It will explicitly exclude checkpoints, datasets, PDFs, backup directories, Python bytecode, compiled shared objects, object files, and build metadata.

<!-- experiment-entry:clean-bireal-four-lutconv-route-20260801 -->
## 2026-08-01 - Clean Phase 2 Bi-Real and Phase 3 four-LUTConv route

### Git Boundary

- The pre-clean source route is preserved on remote branch `archive/four-lutconv-pre-clean-20260801` at commit `dd0c822` (`Archive Bi-Real and LUTConv development snapshot`). Checkpoints and datasets were excluded.
- The clean implementation lives on local branch `route/bireal-four-lutconv-clean`.
- Per the request, this clean branch is intentionally uncommitted, has no upstream, and has not been pushed.

### Active Route

- Phase 1: full-precision complex Bi-Real topology.
- Phase 2: standard complex Bi-Real. CIFAR input still uses `LearnImagBlock`; each residual block is activation -> binary complex convolution -> post-BN -> residual. There is no pre-BN or topology switch.
- Phase 3: identical to Phase 2 except every residual main-path `BinaryComplexConv2d` is replaced by one `FourLUTComplexConv2d`.
- Each complex LUT convolution owns four independent real LUT6 convolutions: `rr(x_r)`, `ii(x_i)`, `ir(x_i)`, and `ri(x_r)`. They combine as `y_r = rr - ii` and `y_i = ir + ri`.
- Each real LUT table uses hard 0/1 forward values from epoch 1 and identity STE backward gradients. Entries use the successful real-LUT bimodal initialization: equal-probability modes centered at -1 and +1, both with standard deviation 0.1.
- Phase 3 defaults to training from scratch. Supplying `CHECKPOINT=...` loads only topology-compatible Phase 2 state; LUT tables retain random bimodal initialization.
- Old pair neurons, analytic comparators, LUT annealing/tau, separate LUT LR, sign-diff capture, and experimental topology/pre-BN switches are removed from the active route.

### Canonical Commands

```bash
GPU_ID=0 WORKDIR=runs/phase2_complex_bireal ./run_phase2.sh
GPU_ID=1 WORKDIR=runs/phase3_four_lutconv ./run_phase3.sh
```

The Phase 3 launcher defaults to 256 epochs, batch size 128, start filters 16, Adam, LR 0.02, the reference Bi-Real milestone schedule, no weight decay or clipping, real-LUT augmentation, label smoothing 0.1, and train/test-only evaluation. All values remain overridable through environment variables.

### Modified Files

- `complexPyTorch/complexBinaryResNet.py`: reduced the model to the fixed Phase 1/2/3 Bi-Real topology and wired Phase 3 to `FourLUTComplexConv2d` only.
- `complexPyTorch/complexLayers.py`: removed archived operator/pair/analytic implementations and retained a minimal hard-STE `LUTBinaryConv2d` plus the four-branch complex wrapper.
- `complexPyTorch/lut_backend.py`: reduced the Python backend to the binary LUTConv CUDA API and autograd wrapper.
- `lut_cuda/setup.py`: reduced extension compilation to `lut_conv_binary_cuda`.
- `training.py`: removed pair conversion, annealing, sign diagnostics, separate LUT optimization, and old CLI switches; added optional shared Phase 2 checkpoint loading for Phase 3.
- `run_phase2.sh`: made the validated standard complex Bi-Real recipe the single Phase 2 launcher.
- `run_phase3.sh`: replaced the pair launcher with the hard four-LUTConv recipe.
- `scripts/audit_active_route.py`: rewrote the reusable route audit around Phase 2 Bi-Real and Phase 3 four-LUTConv structure.
- `tests/test_phase12_route.py`: rewrote tests for the clean CLI, fixed topology, four-branch count, hard-table STE, bimodal initialization, and shared checkpoint loading.
- `EXPERIMENT_LOG.md`: appended this route transition and per-file summary using `scripts/append_experiment_log.py`.

### Deleted Active Files

- Obsolete CUDA sources: `lut_cuda/lut_conv_bafw_cuda_backend.cu`, `lut_cuda/lut_conv_bafw_cuda_backend_top1.cu`, `lut_cuda/lut_conv_bffb_cuda_backend.cu`, `lut_cuda/lut_conv_cuda_backend.cu`, `lut_cuda/lut_cuda_backend.cu`, and `lut_cuda/lut_cuda_backend_binary.cu`.
- Obsolete launchers: `run_phase2_complex_bireal.sh`, `run_phase3_pair_analytic.sh`, and `run_phase3_real_compatible.sh`.
- Obsolete pair analysis: `scripts/analyze_pair_lut_checkpoint.py` and `scripts/analyze_pair_lut_states.py`.
- Obsolete pair tests: `tests/test_pair_lut_phase3.py`.
- Redundant untracked copies already represented by the archive commit: `complexPyTorch/complexBinaryResNet_lut_as_op_backup.py`, `complexPyTorch/complexLayers_lut_as_op_backup.py`, and `complexPyTorch/lut_backend_lut_as_op_backup.py`.

### Verification

- Python syntax compilation passed for the active model, layers, backend, training entry, audit, and tests.
- Shell syntax passed for `run_phase1.sh`, `run_phase2.sh`, `run_phase3.sh`, and `run_training.sh`.
- `conda run -n lut_net python scripts/audit_active_route.py`: passed.
- `conda run -n lut_net python -m unittest tests.test_phase12_route`: 14 tests passed.
- Single `FourLUTComplexConv2d` CUDA forward/backward produced shape `(2, 16, 8, 8)` and non-null input/LUT gradients.
- `run_phase3.sh --help` reached the cleaned training CLI successfully when run with the `lut_net` Python.
- A full Phase 3 one-batch CUDA smoke test was stopped after the custom-kernel process remained alive for over one minute. The single-layer numerical path passed, but full-model runtime/exit behavior still needs confirmation during the next real training launch.

<!-- experiment-entry:grouped-binary-phase3-correction-20260801 -->
## 2026-08-01 - Correction: align Phase 3 with the real grouped-binary LUT kernel

### Reason for the correction

The first cleanup pass temporarily wired the experimental direct-convolution `lut_conv_binary_cuda` backend. Its forward synchronized successfully, but even a `1x16x4x4` backward did not finish within 30 seconds. That implementation is therefore not the training path validated by the real Bi-Real LUT network and was removed before finalizing the clean route.

### Final aligned implementation

- `LUTBinaryConv2d` now exactly follows the real reference dataflow: pad, unfold image patches, reshape to `[batch, locations, groups, lut_num, 6]`, evaluate grouped LUT6 cells, sum the LUT outputs, and reshape to NCHW.
- LUT weights now use the reference layout `[out_channels, lut_num, 64]`.
- Forward LUT entries remain hard 0/1 with identity STE gradients and bimodal +/-1 initialization.
- Autograd now uses `lut_cuda_grouped_binary.forward/backward`, the same grouped-binary extension used by `/home/jliangbr/workspace/LutNet/cifar10_bireal/lut_layer_main_compile.py`.

### File updates

- `complexPyTorch/complexLayers.py`: changed `LUTBinaryConv2d` from direct offset/shift CUDA dispatch to the reference unfold/grouped-LUT layout.
- `complexPyTorch/lut_backend.py`: replaced the experimental direct-convolution wrapper with the minimal reference-compatible grouped-binary autograd function.
- `lut_cuda/setup.py`: changed the only active extension target to `lut_cuda_grouped_binary`.
- `lut_cuda/lut_cuda_backend_binary.cu`: restored the exact grouped-binary source used by the real reference implementation.
- `lut_cuda/lut_conv_binary_cuda_backend.cu`: removed the experimental direct-convolution backend from the clean route.
- `EXPERIMENT_LOG.md`: appended this correction using the reusable log script.

### Verification

- 14 route/unit tests passed after the backend correction.
- Synchronized single-layer CUDA backward completed within the 30-second bound; output shape was `(1,16,4,4)`, with nonzero input and LUT gradients.
- Synchronized full Phase 3 model forward/backward completed within the 60-second bound; output shape was `(1,10)` and all 24 LUTConv layers had gradients.
- This entry supersedes the full-model runtime warning in the immediately preceding cleanup entry.

<!-- experiment-entry:phase3-two-shared-lutconv-20260801 -->
## 2026-08-01 - Correct Phase 3 to two shared LUTConv operators

### Correction

The complex operator must mirror the two parameterized real convolutions in Phase 2, not use four independent LUT parameter sets. The previous four-LUT wrapper over-parameterized the complex convolution and broke real/imaginary weight sharing.

### Final two-LUT formula

For two learned LUTConv operators representing the real and imaginary complex-weight components:

- `conv_r` is reused for both `x_r` and `x_i`.
- `conv_i` is reused for both `x_i` and `x_r`.
- `y_r = conv_r(x_r) - conv_i(x_i)`.
- `y_i = conv_r(x_i) + conv_i(x_r)`.

Calling each module twice builds two autograd paths into the same parameter tensor, so PyTorch accumulates LUT-table gradients and input gradients from both paths. There are two learned LUT parameter sets per complex operator, although software invokes the operators four times. Hardware can time-multiplex the shared tables; fully parallel evaluation would require additional read/replication resources.

### File updates

- `complexPyTorch/complexLayers.py`: retained the user's two-module implementation, renamed it to `TwoLUTComplexConv2d`, clarified cross-term names, and documented real/imaginary LUT sharing.
- `complexPyTorch/complexBinaryResNet.py`: changed Phase 3 to instantiate `TwoLUTComplexConv2d` and corrected the module description.
- `training.py`: corrected the Phase 3 description and fixed Phase 2 checkpoint partial loading. Phase 2 `conv_r/conv_i.weight` keys now explicitly skip the same-named Phase 3 LUT tables instead of raising shape mismatch.
- `run_phase3.sh`: renamed the launcher text and default workdir to `phase3_two_lutconv`.
- `tests/test_phase12_route.py`: changed structural expectations from four to two LUTConv modules, added an exact analytic cross-term test, and retained checkpoint initialization coverage.
- `scripts/audit_active_route.py`: changed the active-route contract to exactly two LUTConv modules per complex residual operator.
- `EXPERIMENT_LOG.md`: appended this correction through the reusable log script.
- Local branch renamed from `route/bireal-four-lutconv-clean` to `route/bireal-two-lutconv-clean`; it remains uncommitted and unpushed.

### Verification

- 15 unit tests passed.
- The analytic test verifies `conv_r=2I`, `conv_i=3I` produces `(2x_r-3x_i) + j(2x_i+3x_r)` exactly.
- Phase 2 checkpoint partial loading loaded 65 shared tensors and preserved 12 randomly initialized LUT tables in the six-operator test model.
- The route audit passed and confirms two shared LUTConv modules per complex operator.
- Synchronized full CUDA forward/backward passed with output shape `(1,10)`; all 12 LUT tables received gradients.

<!-- experiment-entry:current-route-method-document-20260802 -->
## 2026-08-02 - Rewrite current technical route as a concise method document

### Documentation update

- Replaced the obsolete Pair-LUT technical-route document with a concise description of the current method.
- The document now explains only the active Phase 1 -> Phase 2 -> Phase 3 progression.
- It defines the two shared LUTConv operators and their complex cross-computation equations.
- It summarizes LUT6 local grouping, hard-forward STE training, and the time-multiplexed or parallel hardware interpretation.
- Experimental history, parameter sweeps, retired branches, commands, and implementation-level details were intentionally omitted.

### Modified files

- `CURRENT_TECHNICAL_ROUTE.md`: rewritten as the high-level method document for the current complex Bi-Real to two-LUTConv route.
- `EXPERIMENT_LOG.md`: recorded this documentation change through `scripts/append_experiment_log.py`.

<!-- experiment-entry:phase3-exact-zero-one-lut-input-20260802 -->
## 2026-08-02 - Correct Phase 3 LUT inputs from signed values to exact 0/1 bits

### Problem identified

Phase 2 Bi-Real activations use signed binary values `{-1,+1}`, while the grouped LUT6 CUDA kernel interprets each address input as a hardware bit `{0,1}` using a `>0.5` comparison. Although `-1/+1` happened to select the same forward addresses as `0/1`, directly passing signed values was not an equivalent training formulation:

- The CUDA finite-difference backward is the derivative with respect to a `0/1` bit coordinate.
- The missing conversion `b=(s+1)/2` omitted its factor `1/2` from the chain rule.
- `sign(0)=0` was not an exact signed bit and relied implicitly on the CUDA threshold.

### Correction

Phase 2 remains unchanged and continues to use signed complex Bi-Real activation.

Phase 3 now uses `BinaryComplexBitActivation`:

- Hard forward: each real and imaginary component is exactly `1[x>0]`, including an exact zero bit when `x=0`.
- Backward proxy: Bi-Real signed activation followed by `(s+1)/2`, so the gradient is `max(1-|x|,0)`, exactly half the signed Bi-Real proxy.
- The two shared LUTConv operators therefore receive true `0/1` hardware address bits and the CUDA input finite difference is connected with the correct chain-rule scale.

### Modified files

- `complexPyTorch/complexLayers.py`: added `BinaryComplexBitActivation` with exact 0/1 hard forward and scaled Bi-Real backward.
- `complexPyTorch/complexBinaryResNet.py`: Phase 3 now uses bit activation; Phase 2 keeps signed activation.
- `training.py`: clarified the Phase 3 description as a 0/1-input two-LUTConv network.
- `run_phase3.sh`: added an explicit 0/1 LUT-input startup message.
- `tests/test_phase12_route.py`: added exact value and gradient tests and Phase 2/3 activation-type assertions.
- `scripts/audit_active_route.py`: added the signed-Phase-2 versus bit-Phase-3 route contract.
- `CURRENT_TECHNICAL_ROUTE.md`: documented the signed-to-bit encoding and its backward chain factor.
- `EXPERIMENT_LOG.md`: appended this correction with the reusable log script.

### Verification

- 16 unit tests passed.
- The exact bit test verifies forward `[0,0,0,1,1]` for inputs `[-2,-0.5,0,0.5,2]` and gradients `[0,0.5,1,0.5,0]`.
- The active-route audit passed and distinguishes Phase 2 signed activation from Phase 3 bit activation.
- Full synchronized CUDA forward/backward passed; a pre-forward hook observed the first LUTConv input values as exactly `[0.0,1.0]`, and all 12 LUT tables received gradients.

<!-- experiment-entry:real-lut-reference-gap-audit-20260803 -->
## 2026-08-03 - Real-LUT reference numerical and gradient audit

Files: scripts/analyze_lut_reference_gap.py, reports/lut_reference_gap_current.md, reports/lut_reference_gap_current.json, and this EXPERIMENT_LOG.md entry. Added a reusable checkpoint-and-batch audit for Phase 3 LUT input bits, per-call LUT sums, complex pre-BN ranges, BN outputs, LUT margins, gradient norms, and shared-path gradient interaction. Audited runs/phase3_four_lutconv/chkpts/Bestmodel_phase3.pt (best test accuracy 0.7750). Confirmed exact 0/1 activation inputs and the same LUT6 CUDA/sum contract as the real reference. Found complex-only asymmetric pre-BN aggregation: real=A-B is centered while imag=C+D has means near L (23.4, 48.3, 96.2 by depth). First-layer shared paths are nearly orthogonal, with combined/independent gradient ratios 0.9885-1.0118, so severe first-layer cancellation is not supported. Found stronger depth attenuation: mean LUT gradient RMS is 7.83e-3 in stage2, 2.44e-4 in stage3, and 7.64e-5 in stage4. Mean abs(logit) has drifted to 2.76/2.93/3.26, leaving only 2.3%-3.2% within abs(logit)<0.1. Also identified recipe mismatches: current run uses Adam LR 0.02 with milestone drops, while the reference source defaults to LR 0.001 with linear decay; current negative-mode init std is 0.1 versus reference 0.2. Priority: reproduce optimizer schedule first, then test current-route BN and deeper gradient paths; signed LUT decoding is lower priority because train-mode BN mostly absorbs its affine shift/scale.

<!-- experiment-entry:align-phase3-reference-scheduler-20260803 -->
## 2026-08-03 - Align Phase 3 scheduler and explain complex BN/deep LUT gradients

Modified files: training.py, run_phase3.sh, tests/test_phase12_route.py, scripts/analyze_lut_reference_gap.py, CURRENT_TECHNICAL_ROUTE.md, reports/lut_reference_gap_current.md, reports/lut_reference_gap_current.json, and this EXPERIMENT_LOG.md entry.

Aligned Phase 3 scheduling with the local real-domain LUT reference. The schedule name `bireal_reference` now uses the same per-epoch linear rule `lr(epoch)=base_lr*(1-epoch/num_epochs)`; the former drops at epochs 90/140/180/220 remain available as `multistep` for historical reproducibility. Changed the Phase 3 launcher default Adam LR from 0.02 to the reference default 0.001. Added a regression test that checks exact rates at epochs 0, 128, and 255 and equivalence with `linear`.

Clarified in the generated numerical audit that the 77.50% checkpoint is historical and used the former 0.02 milestone recipe. The current launcher now uses the aligned recipe. Updated the active-route document accordingly.

BN analysis: covariance complex BN first subtracts the per-channel complex mean, so the LUT composition's imaginary DC offset near L is removed directly during training. It then estimates the 2x2 real/imag covariance, applies its inverse square root, and finally applies a learned symmetric 2x2 affine transform plus complex bias. This removes affine offset/scale in the forward pass but does not erase batch-statistic noise, eps effects, running-statistic mismatch, or the altered backward Jacobian.

Gradient interpretation: the reported values are per-LUT-entry gradient RMS, not activation gradient magnitude. Deeper stages have 16x fewer spatial sites (32x32 to 8x8), up to 4x more LUT groups per output, 4x more output channels, and 64 mutually exclusive addresses per LUT. Each deep truth-table entry is therefore hit much less often, explaining why its gradient can be much smaller even though the layer is closer to the loss. Verification: 17 route unit tests passed; run_phase3.sh passed bash syntax validation; the CUDA numerical report regenerated successfully.

<!-- experiment-entry:progressive-layerwise-lut-flow-20260803 -->
## 2026-08-03 - Progressive float-to-hard layer-wise LUT training flow

Modified files: complexPyTorch/complexLayers.py, training.py, run_phase3.sh, tests/test_phase12_route.py, scripts/audit_active_route.py, CURRENT_TECHNICAL_ROUTE.md, and this EXPERIMENT_LOG.md entry.

Added an optional Phase 3 progressive LUT quantization flow without changing the default hard-from-start route. LUT activation addresses remain exact 0/1 throughout. During soft stages, truth-table entries are continuous values computed as (tanh(tau * logit) + 1) / 2 and are read directly by the existing grouped binary-address CUDA kernel. This preserves exact O(1) address lookup while making both table values and finite-difference activation gradients continuous.

The progressive state machine is: float warmup, global geometric temperature annealing, layer-wise hardening, then fully-hard fine-tuning. Default 256-epoch parameters are 20 warmup epochs at tau 0.5, 100 annealing epochs from tau 0.5 to 8.0, 72 layer-hardening epochs, and 64 fully-hard epochs. The 18 complex residual operators are hardened as complete units, switching their real and imaginary LUTConv branches together. Default order is input-to-output so downstream soft layers can adapt to upstream quantization; output-to-input is also configurable.

Hard layers use hard 0/1 forward plus identity STE, avoiding saturated tanh gradients after commitment. Soft layers stay at tau_max during layer-wise hardening. Bestmodel_phase3.pt selection is gated by hardware_ready, so soft or partially-hard epochs cannot overwrite the deployable best checkpoint; Lastmodel_phase3.pt continues to capture every epoch. Progressive mode rejects torch.compile to avoid repeated graph specialization as layer modes change.

Launcher controls: LUT_TRAINING_FLOW, LUT_FLOAT_WARMUP_EPOCHS, LUT_ANNEAL_EPOCHS, LUT_LAYER_HARDENING_EPOCHS, LUT_TAU_MIN, LUT_TAU_MAX, and LUT_HARD_ORDER. Existing hard flow remains the default.

Verification: 19 route unit tests passed. Tests cover the soft differentiable table and the warmup->anneal->one-complex-operator-at-a-time->fully-hard state machine. The active-route audit and run_phase3.sh syntax check passed. A CUDA smoke test on GPU 2 observed soft non-integer LUT sums (maximum distance from an integer 0.363) with nonzero table/input gradients, followed by strictly integer hard sums with nonzero table/input gradients using the same binary-address backend.

<!-- experiment-entry:pruning-three-stage-activation-lut-20260805 -->
## 2026-08-05 - Pruning-style three-stage activation and LUT schedule

### 三阶段训练策略

Phase 3 progressive flow 改为 pruning 风格的严格三阶段状态机：

1. `temperature_anneal`：activation 与 LUT 均保持 soft，tau 按 `tanh(logit * tau)` 的语义从 `lut_tau_min` 线性增大到 `lut_tau_max`。
2. `layer_hardening`：tau 固定在 `lut_tau_max`，按网络顺序逐个 hard 复数 block；一个 block 的 activation、实部 LUT 和虚部 LUT 同时切换。
3. `fully_hard`：所有 activation 和 LUT 使用硬前向与 STE，继续训练 BN、LUT 参数和其余网络参数。只有此阶段标记为 hardware-ready 并参与最佳 checkpoint 选择。

### 修改文件

- `complexPyTorch/complexLayers.py`：为 `BinaryComplexBitActivation` 增加 tau/hard 状态；soft 前向使用 `sigmoid(2*tau*x)`，与 LUT 的 `(tanh(tau*x)+1)/2` 完全等价；hard 前向继续使用精确 0/1 和原 Bi-Real STE。
- `training.py`：实现 activation 与 `TwoLUTComplexConv2d.conv_r/conv_i` 的同步三阶段调度；改用线性递增 tau；删除独立 warmup stage；旧 warmup CLI 名保留为 anneal 参数兼容别名；所有旧 `LUTBinaryConv2d` 路线引用改为当前 `LUTFPConv2d`。
- `tests/test_phase12_route.py`：更新 LUT 类型断言与三阶段边界测试；增加 sigmoid soft activation 数值和梯度测试；保留最终 hard Bi-Real 梯度测试。
- `EXPERIMENT_LOG.md`：通过可复用日志脚本追加本记录。

### 验证

- 三个修改代码文件通过 `py_compile`。
- 全量 diff 检查仅报告用户此前 LUTFP/LUT backend 集成区域已有的行尾空格，本轮新增行没有格式错误。
- 定向 unittest 在当前机器导入训练栈时持续高 CPU 且长时间无输出，已主动终止，未遗留后台测试进程。
- `lut_backend.py` 与 CUDA kernel 均未修改。

<!-- experiment-entry:phase3-progressive-nan-epoch35-20260805 -->
## 2026-08-05 - Phase 3 progressive FP-LUT NaN diagnosis and safeguards

Run: `runs/phase3_lutfp_progressive_tau1_2_a40_h90_e256`

### Observation

- Finite through epoch 34; best test accuracy 69.31% at epoch 32.
- First non-finite metrics at epoch 35 with `tau=1.87179`, `hard_operators=0/18`, LR `0.0008671875`. Failure occurred in all-soft annealing before layer hardening while LR was decreasing.
- Epoch-43 Last checkpoint has 136 non-finite model tensors and about 5.716 million non-finite model elements. LUT, BN, frontend, classifier, and Adam state are contaminated and cannot be resumed.

### Files and fixes

- `complexPyTorch/complexLayers.py`: changed soft LUT annealing from `tanh(logits / tau)` to reference-consistent `tanh(logits * tau)`, so LUT and activation harden in the same direction.
- `training.py`: fail fast on non-finite loss and report batch/contaminated parameters; norm clipping now rejects non-finite gradient norms before optimizer update.
- `run_phase3.sh`: expose `CLIPNORM` and `CLIPVAL`; both were previously hard-coded to zero.
- `tests/test_phase12_route.py`: regression test verifies larger tau sharpens LUT entries; focused and complete active-route tests pass.
- `scripts/analyze_training_run.py`: parse nan/inf and report the first non-finite epoch.
- `scripts/analyze_checkpoint_nonfinite.py`: reusable model/optimizer checkpoint tensor scanner.

### Next experiment

Restart from the clean Phase 2 checkpoint in a new workdir with the 40/90 progressive schedule and `CLIPNORM=5`. Never load the poisoned Phase 3 Last checkpoint. A recurrence will now stop at the first affected batch while preserving the prior finite checkpoint.

<!-- experiment-entry:phase3-mul-tau-clip5-stopped-epoch18-20260805 -->
## 2026-08-05 - Phase 3 corrected-tau run stopped during epoch 18

Run: `runs/phase3_lutfp_progressive_mul_tau1_2_a40_h90_clip5_e256`

- A first launch at 09:35 exited before completing epoch 1. A second launch at 14:51 trained normally through epoch 17.
- Each completed epoch took about 592-594 seconds. Test accuracy increased from 37.92% at epoch 1 to 60.31% at epoch 17. No NaN/Inf was logged.
- The log stopped after entering epoch 18 at `tau=1.4359`, LR `0.00093359375`, with all 18 operators still soft. At inspection time no project training process remained on any GPU; this was process termination, not a still-running deadlock.
- `Lastmodel_phase3.pt` is epoch 17. All 228 model tensors and all 276 optimizer tensors are finite, so this checkpoint is valid for recovery.
- Persistent `train.txt` does not capture process stderr, and kernel logs are unavailable to this user. The remaining evidence cannot distinguish a CUDA extension failure, shell signal, or external kill. A resumed run should redirect both stdout and stderr to a console log and retain the epoch-17 checkpoint.

<!-- experiment-entry:phase3-mul-tau-clip5-epoch18-gradient-correction-20260805 -->
## 2026-08-05 - Correction: epoch 18 stopped on non-finite gradient norm

This entry corrects the earlier interpretation that the epoch-18 disappearance could only be an external termination. The terminal traceback was not present in persistent `train.txt` but shows the exact stop condition.

- At epoch 18 (`tau=1.4359`, LR `0.00093359375`, 0/18 hard operators), forward loss was still finite.
- After `loss.backward()`, `torch.nn.utils.clip_grad_norm_(..., error_if_nonfinite=True)` found a non-finite global L2 gradient norm and raised before `optimizer.step()`.
- Gradient clipping did not cause the failure; it detected and contained it. Setting `error_if_nonfinite=False` would allow contamination and must not be used.
- The epoch-17 model and optimizer checkpoint remain fully finite.
- Compared with the former divide-tau experiment, which first became non-finite at epoch 35 around tau 1.87, multiply-tau failed at epoch 18 around tau 1.44. This strongly indicates that sharpening LUT entries increases finite-difference input-gradient gain and destabilizes the deep soft-LUT chain earlier.
- The next diagnostic change should report the exact parameter names and per-group gradient maxima when the norm becomes non-finite; Phase 3 resume support would avoid replaying 17 epochs solely to reproduce the failure.

<!-- experiment-entry:phase3-nonfinite-gradient-debug-20260805 -->
## 2026-08-05 - Add non-finite gradient explosion diagnostics

### Purpose

The corrected multiply-tau run failed at epoch 18 because the post-backward global gradient norm was non-finite. Add enough diagnostics to distinguish true NaN/Inf gradient elements from float32 norm-reduction overflow and identify the responsible parameter tensors.

### File changes

- `training.py`: track the maximum pre-clip global gradient norm and batch for every completed epoch. When `clip_grad_norm_` detects a non-finite norm, scan gradients before any optimizer update and atomically write `WORKDIR/debug/nonfinite_grad_epochXXXX_batchXXXX.json`. The report includes epoch/batch, finite forward loss and output magnitude, diagnosis class, all non-finite gradient tensors with NaN/+Inf/-Inf counts, float64 L2 norms and maximum magnitudes, the 20 largest still-finite gradients, and every LUT/activation tau and hard state. The raised error now points directly to this report.
- `tests/test_phase12_route.py`: add a regression test that creates Inf/NaN gradients and verifies the generated report identifies the parameter and element counts.

### Interpretation

If `diagnosis=nonfinite_gradient_elements`, inspect the listed names to locate the contaminated backward path. If `diagnosis=float32_global_norm_overflow_with_finite_elements`, individual gradients remain finite and the problem is overflow in norm aggregation caused by extremely large finite values; the largest-finite list identifies the amplification source.

### Verification

Python compilation passed. All 22 tests in `tests.test_phase12_route` passed. The protected `complexPyTorch/lut_backend.py` and CUDA kernels were not modified.

<!-- experiment-entry:phase3-mul-tau-debug-epoch7-batch57-20260808 -->
## 2026-08-08 - Debug run: localized NaN backward at epoch 7 batch 57

Run: `runs/phase3_lutfp_progressive_mul_tau_debug_gpu1`

### Result

- The run completed six epochs and stopped during epoch 7, batch 57. The last completed test accuracy was 49.14%. The process is no longer running.
- This invocation used batch size 256, not 128. It remained in the all-soft stage; failure tau was 1.153846 and 0/18 operators were hard.
- Forward was finite at failure: loss 1.77758 and maximum output magnitude 4.82835.
- Pre-clip global gradient norms in epochs 1-6 were 3.94664, 3.96013, 4.42090, 3.92388, 3.79772, and 4.29326. There was no increasing trend and ordinary clipping had not been persistently active.

### Gradient report

Report: `runs/phase3_lutfp_progressive_mul_tau_debug_gpu1/debug/nonfinite_grad_epoch0007_batch0057.json`

- Diagnosis is `nonfinite_gradient_elements`, not float32 norm-reduction overflow.
- Fourteen parameter-gradient tensors contain NaNs. They are confined to `stage2.0` and all modules before it: `stage2.0.conv.conv_r/i.weight`, `stage2.0.bn_post.weight/bias`, stem `bn1/conv1`, and `learn_imag`.
- All parameter gradients from `stage2.1` onward remain finite. Their largest absolute element is only about 0.218. The finite remainder has a float64 global L2 norm of about 1.051.
- Therefore this is a sudden localized NaN generated at the shallow backward boundary, not gradual whole-network gradient explosion.

### Root-cause narrowing

Backpropagation traverses deeper blocks before shallower blocks. Finite `stage2.1` parameter gradients but NaN `stage2.0` and earlier gradients place the first invalid value between the `stage2.1` input-gradient calculation and the `stage2.0` post-BN backward. The two primary candidates are the FP-LUT custom backward returning a NaN input gradient from `stage2.1`, or the covariance complex BN backward in `stage2.0`.

The finite epoch-6 checkpoint does not show persistent covariance singularity: `stage2.0.bn_post.running_covar` has determinant minimum about 3.79 and maximum absolute correlation about 0.301. A single bad batch covariance remains possible, but the running statistics do not support general BN collapse.

### Next diagnostic

Use PyTorch anomaly detection or targeted backward hooks around `stage2.1.conv`, `stage2.1.act`, and `stage2.0.bn_post`. This should identify whether `LUTConvFP32FunctionBackward` first returns NaN in its input-gradient output or complex BN creates it. A naive post-BN control run is also a clean A/B test for the covariance-BN hypothesis.

<!-- experiment-entry:phase3-all-stage-backward-hooks-20260808 -->
## 2026-08-08 - Add ordered backward hooks across all residual stages

### Goal

Capture the first invalid gradient boundary regardless of which residual stage fails in future progressive FP-LUT experiments.

### File changes

- `training.py`: added optional `BackwardHookRecorder`. With `--debug-backward-hooks`, it registers full backward hooks on every `BinaryComplexBitActivation`, `LUTFPConv2d`, `TwoLUTComplexConv2d`, covariance complex BN, and naive complex BN module under `stage2`, `stage3`, and `stage4`. Each failing batch report now contains the ordered backward trace with per-module `grad_output` and `grad_input` shape, dtype, finite/NaN/Inf counts, finite maximum magnitude, and float64 L2 norm. The raised exception prints the first hooked non-finite module, type, and backward sequence number. Records reset each batch and are serialized only when the existing non-finite global-norm guard triggers.
- `run_phase3.sh`: added `DEBUG_BACKWARD_HOOKS=1`, which forwards `--debug-backward-hooks`.
- `tests/test_phase12_route.py`: added a regression test proving stage hooks register and identify the first non-finite backward record.

### Coverage and verification

The full Phase 3 model registers 92 hooks: 30 in stage2, 31 in stage3, and 31 in stage4. Shared LUT branches may generate multiple ordered records per batch, preserving real/imaginary cross-path call information. Python compilation and shell syntax checks passed. All 23 active-route tests passed. The protected `complexPyTorch/lut_backend.py` and CUDA kernels were not modified.

### Runtime note

This mode performs finite-value checks at every hooked backward boundary and is intended for diagnosis; it will be slower than ordinary training. Normal logs remain concise. On failure, inspect `WORKDIR/debug/nonfinite_grad_epochXXXX_batchXXXX.json`, especially `backward_hooks.first_nonfinite` and `backward_hooks.records`.

<!-- experiment-entry:phase3-stage-block-gradient-probes-20260808 -->
## 2026-08-08 - Correction: replace intrusive LUT hooks with stage-block probes

### Correction

The previous all-module implementation was not safe for the custom CUDA LUT path. Both `register_full_backward_hook` and tensor hooks placed directly on `LUTFPConv2d` boundaries made the otherwise stable configuration produce NaN at epoch 1 batch 0. The apparent first failure at `stage2.5.conv.conv_i` was instrumentation-dependent and must not be treated as a model root-cause result.

### A/B evidence

- Hooks enabled inside LUT/activation/BN modules: epoch 1 batch 0 immediately produced non-finite gradients.
- Identical seed, checkpoint, batch size, schedule, and GPU with hooks disabled: ran for the full 90-second smoke window without an error.
- The failure is therefore caused by intrusive observation of the custom CUDA backward path, not by the baseline configuration naturally failing on its first batch.

### File changes

- `training.py`: replaced internal custom-operator hooks with output-gradient probes at every residual block under `stage2`, `stage3`, and `stage4`. Hook callbacks only retain detached gradient references; finite/NaN/Inf statistics are computed after backward and only when the existing non-finite guard requests a report. This avoids wrapping LUT tensors and avoids CUDA synchronization inside custom backward callbacks. The full model exposes 18 block-boundary probes.
- `EXPERIMENT_LOG.md`: recorded this correction so the earlier 92-hook implementation is not reused or interpreted as valid evidence.

### Verification

- Python compilation passed and the focused NaN-capture regression test passed.
- A real GPU1 smoke run with the corrected 18 block probes survived 90 seconds and did not create a non-finite report; the old implementation failed at batch 0 within seconds.
- `complexPyTorch/lut_backend.py` and all CUDA kernel files were not modified.

<!-- experiment-entry:phase3-stage2-fplut-invalid-lanes-20260808 -->
## 2026-08-08 - Phase 3 stage2 FP-LUT backward NaN root cause

Run: `runs/phase3_lutfp_mul_tau_stagehooks_bs256_gpu1`

### Failure

- Epochs 1 and 2 completed normally, reaching test accuracies 35.88% and 41.19%.
- Epoch 3 failed at batch 148 while all 18 operators were soft, tau was 1.051282, and LR was 0.0009921875.
- Forward remained finite: loss 1.87293 and output maximum magnitude 4.52519. This is a backward-only failure.
- The epoch-2 `Lastmodel_phase3.pt` is safe: all 228 model tensors and all 276 optimizer tensors are finite.

### Gradient localization

Report: `runs/phase3_lutfp_mul_tau_stagehooks_bs256_gpu1/debug/nonfinite_grad_epoch0003_batch0148.json`

- Block-boundary gradients are fully finite from `stage4.5` through `stage2.3`.
- The gradient at the output of `stage2.2`, which is the input gradient returned through `stage2.3`, is the first invalid boundary: 2,944 of 4,194,304 complex elements are NaN.
- The contamination then expands to every element at `stage2.1` and `stage2.0`.
- Parameters in `stage2.3` remain finite, including both LUT weight gradients and complex-BN gradients. Parameter NaNs begin at `stage2.2` and propagate toward the stem.
- Epoch-2 `stage2.3` LUT logits are ordinary (absolute maximum about 1.43), and its running complex covariance is well conditioned (determinant 7.41 to 15.56, absolute correlation at most 0.246). This rules out persistent BN singularity or runaway logits.

### Root cause

The evidence identifies the FP-LUT CUDA input-gradient path in `stage2.3`. Stage 2 has 16 output channels, while `lut_conv_fp32_backward_kernel` reduces over a fixed 32-lane tile. Lanes 16-31 have no valid output channel, so `dy=0`, but their `s_w[..., lane]` entries are not initialized. Those invalid lanes still compute `dx_accum` and participate in the full-warp reduction used for `grad_x`. If stale shared memory contains NaN, IEEE arithmetic preserves it through `0 * NaN`, and the warp reduction contaminates an otherwise finite input gradient. The `grad_w` path is guarded by valid/nonzero `dy`, explaining why LUT weight gradients remain finite while `grad_x` fails. Stages 3 and 4 use 32/64 output channels and therefore do not expose the partial-tile condition.

This is an intermittent CUDA partial-output-tile bug, not gradient explosion caused by tau, LR, activation annealing, or complex BN. The protected `complexPyTorch/lut_backend.py` and CUDA files were inspected read-only and were not modified.

### File changes

- `scripts/analyze_nonfinite_gradient_report.py`: added a reusable parser that summarizes failure metadata, contaminated parameter names, and ordered residual-block gradient boundaries from future non-finite JSON reports.
- `EXPERIMENT_LOG.md`: recorded this experiment and root-cause analysis.

<!-- experiment-entry:real-bireal-vs-fused-fp-kernel-20260808 -->
## 2026-08-08 - Why the real Bi-Real LUT route did not trigger the FP-kernel NaN

The real CIFAR Bi-Real LUT network also contains a 16-channel first residual stage, so channel width alone does not explain why it trained without NaN. The decisive difference is the operator path.

- The real model's `BasicBlock` uses `lut_conv_group`, performs `unfold/img2col`, reshapes inputs to explicit LUT groups, and calls `LUT6Function -> lut_cuda_grouped_binary`.
- That grouped binary kernel organizes work over explicit LUT/output indices and does not use the current fused convolution kernel's fixed 32-output-channel warp reduction.
- The real Bi-Real block also sets each LUT kernel to hard mode at construction.
- The current complex progressive route uses `LUTFPConv2d -> LUTConvFP32Function -> fused_lut_conv_bw`. Its stage2 output width is 16, exposing a partial 32-lane output tile whose invalid lanes can contaminate `grad_x`.

Therefore the real model's successful training does not contradict the current diagnosis: it did not exercise the faulty fused FP partial-output-tile backward path. The next required engineering step is to correct and directly test the CUDA FP kernel for output channel counts below or not divisible by 32 before resuming long progressive training. No kernel or backend file was modified during this comparison.

<!-- experiment-entry:fplut-valid-oc-gradx-mask-20260808 -->
## 2026-08-08 - Single-line FP-LUT partial-tile CUDA fix

### Source change

Only one source line was changed in `lut_cuda/lut_conv_cuda_backend.cu`: the value entering the `grad_x` warp reduction is now `valid_oc ? dx_accum[k] : 0.0f`. No other source line, LUT backend wrapper, training file, or model file was changed for this fix.

### Build and deployment

- Rebuilt the CUDA extensions in the `lut_net` conda environment. CUDA 13.2 reported a minor-version mismatch with the CUDA 13.0 used to build PyTorch, but compilation and linking completed successfully.
- Deployed only the rebuilt `lut_conv_fp32_cuda` binary to the environment's site-packages path used by training.
- Verified that Python loads `/home/jliangbr/miniconda3/envs/lut_net/lib/python3.10/site-packages/lut_conv_fp32_cuda.cpython-310-x86_64-linux-gnu.so`.

### CUDA verification

On GPU1, tested `LUTConvFP32Function` with output channel counts 8, 16, 24, 32, 40, and 64. Each case ran 100 independent forward/backward iterations. All 600 iterations produced finite forward outputs, finite `grad_x`, and finite `grad_w`. This covers partial output tiles below 32, non-multiples of 32, and complete tiles.

The failed Phase 3 workdir must not be resumed from a poisoned batch. Restart from the Phase 2 checkpoint in a new workdir using the same progressive schedule.

<!-- experiment-entry:phase3-remove-hooks-constant-lr002-20260810 -->
## 2026-08-10 - Remove backward hooks and use constant 0.02 Phase 3 LR

With the FP-LUT CUDA partial-tile fix eliminating the observed NaN, remove the temporary runtime gradient instrumentation and restore ordinary training execution.

### File changes

- `training.py`: removed `BackwardHookRecorder`, block-output tensor hook registration, per-batch hook reset, hook snapshots in non-finite reports, the hook-specific exception text, model attachment logic, and the `--debug-backward-hooks` CLI option. The existing parameter-gradient finite checks, atomic JSON failure report, gradient clipping, and tau/hard-state diagnostics remain intact.
- `run_phase3.sh`: removed the `DEBUG_BACKWARD_HOOKS` environment switch and CLI forwarding. Changed Phase 3 defaults from LR 0.001 with `bireal_reference` linear decay to LR 0.02 with the `constant` schedule.
- `tests/test_phase12_route.py`: removed the obsolete backward-hook regression test. All other route, quantization, scheduler, checkpoint, and failure-report tests remain.
- `EXPERIMENT_LOG.md`: recorded the cleanup, resulting runtime behavior, and verification.

### Learning-rate behavior

The Adam optimizer uses `args.lr` for every parameter group, and `set_optimizer_lr` applies the scheduler result to every group. With the new defaults, LUT entries, complex BN parameters, stem, projections, and classifier all use LR 0.02 for every epoch. Environment overrides remain available only when explicitly supplied.

### Verification

- No hook recorder, hook registration, debug-hook CLI flag, or shell debug-hook switch remains in `training.py`, `run_phase3.sh`, or the active tests.
- Python compilation and shell syntax checks passed.
- All 22 tests in `tests.test_phase12_route` passed.
- Direct scheduler verification confirmed all 256 epochs return exactly 0.02 under the constant schedule.

<!-- experiment-entry:phase3-progressive-lr002-structural-vs-optimization-20260811 -->
## 2026-08-11 - Interpret current progressive LUT loss: representation versus optimization

- Run: `runs/phase3_lutfp_progressive_lr002_constant_gpu1/logs/train.txt`. The file contains two invocations; this analysis uses the second/current invocation started at 2026-08-10 15:35.
- Reference: the source Phase 2 checkpoint reached 85.05% best test accuracy.
- Observed trajectory: Phase 3 reached 80.76% at epoch 20 while all 18 operators were still soft; at epoch 40 (`tau=2`, 0/18 hard) test accuracy was 77.27%. During progressive hardening it was 78.30% at 12/18 hard, 74.32% at 14/18, 68.25% at 15/18, and about 70.28% at 16/18 as of the inspected log.
- Optimization evidence: pre-clip gradient norms rose sharply late in hardening (including about 182.3 and 284.8) while `clip_grad_norm=5`. Therefore late-stage updates are strongly clipped and the hardening transition also creates a difficult, unstable optimization problem.
- Structural interpretation: each original six-term binary local accumulation has seven possible levels (`-6,-4,-2,0,2,4,6`). A hard LUT emits only one bit and collapses those seven levels to two. This is a real local output-bandwidth/information bottleneck, and the accuracy decline tracking the hard-operator count is strong evidence that hard conversion causes genuine representational loss.
- Important qualification: it is not correct to say optimization is irrelevant. Structural loss and optimization difficulty coexist. Also, the current Phase 3 loads shared Phase 2 tensors while LUT tables use bimodal initialization, so the soft-stage gap includes initialization shock and does not isolate the architectural loss.
- Proposed isolation experiment: (A) initialize every soft LUT exactly from its corresponding Phase 2 local sum using an affine normalization and recalibrate BN, then test immediate equivalence; (B) hard-project that same checkpoint without training to measure pure representational loss; (C) fine-tune the hard model to measure how much of that loss is recoverable. This separates initialization, hard representation, and optimization effects.
- Files changed for this analysis: only `EXPERIMENT_LOG.md`; no model, training, LUT backend, or CUDA source was modified.

<!-- experiment-entry:review-binary-twolut-rollback-20260811 -->
## 2026-08-11 - Review binary rollback in TwoLUTComplexConv2d

- Scope: reviewed the new `LUTBinaryConv2d` and `TwoLUTComplexConv2d` implementation at the end of `complexPyTorch/complexLayers.py`, plus the current Phase 3 model, scheduler, checkpoint loader, and optimizer integration. No source code was changed.
- Correct part: sharing `conv_r` for `rr/ri`, sharing `conv_i` for `ii/ir`, and returning `(rr-ii) + j(ri+ir)` preserves the intended two-operator complex cross pattern.
- Blocking scheduler mismatch: `configure_lut_training_flow()` writes `branch.tau` and `branch.hard`, but `LUTBinaryConv2d` defines neither buffer. Both hard and progressive Phase 3 flows will fail when configuration is applied.
- Blocking soft-input mismatch: progressive `BinaryComplexBitActivation` emits values strictly between 0 and 1, while `LUTBinaryConvFunction.forward()` converts inputs with `x.to(torch.int8)`. Almost every soft activation therefore becomes zero. A binary-kernel path must keep activation hard from the start, or use a non-binary kernel during the soft stage.
- Blocking checkpoint mismatch: `load_phase3_shared_checkpoint()` recognizes only `LUTFPConv2d` as a LUT layer. New `LUTBinaryConv2d.weight` tensors are therefore treated as shared Phase 2 convolution weights; their shapes differ and the loader raises an incompatible-shape error instead of retaining bimodal LUT initialization.
- Coverage issue: `LUTBinaryConv2d.lut_num` uses floor division. For the first stage with 11 channels and a 3x3 kernel, 99 positions become only 16 LUTs, covering 96 positions and dropping three. `ceil` alone would prevent dropping but the extra LUT inputs also need an explicit neutral-padding policy rather than accidental wraparound.
- Integration inconsistencies: LUT layer counting, gradient diagnostics, and no-weight-decay classification still recognize `LUTFPConv2d` only. Training may still update binary LUT weights through the generic decay group, but behavior and reporting no longer match the intended LUT configuration.
- Static syntax check passed for `complexLayers.py`, `complexBinaryResNet.py`, and `training.py`; this does not cover the runtime interface failures above.
- Files changed for this review: only `EXPERIMENT_LOG.md` was appended through `scripts/append_experiment_log.py`.

<!-- experiment-entry:phase3-fully-hard-binary-rollback-20260811 -->
## 2026-08-11 - Restore Phase 3 to fully hard binary Bi-Real/LUT training

- Goal: fully roll Phase 3 back from floating/annealed LUT training to binary training aligned with the real Bi-Real LUT reference.
- `complexPyTorch/complexLayers.py`: `BinaryComplexBitActivation` now explicitly performs the original Bi-Real signed activation first, forces the zero boundary to the `-1` branch, and only then applies `(s + 1) / 2`. The LUT kernel therefore receives exact `{0,1}` bits while backward remains the Bi-Real proxy with the expected one-half chain factor. `LUTBinaryConv2d` uses hard `{0,1}` LUT entries with identity STE and the binary CUDA backend. It now rejects convolution layouts whose per-group logical input count is not divisible by six instead of silently dropping positions; this also respects the fixed LUT-count dispatches in the existing CUDA kernel. `TwoLUTComplexConv2d` uses two shared binary LUT branches and retains `(rr-ii) + j(ri+ir)`.
- `training.py`: all active Phase 3 integration now recognizes `LUTBinaryConv2d` rather than `LUTFPConv2d`. Phase 2 checkpoint loading excludes binary LUT tables and keeps their bimodal initialization; LUT parameters are put in the no-weight-decay group; gradient diagnostics and layer counts report binary LUT branches. The soft anneal/layer-hardening scheduler and its validation were removed. Every Phase 3 epoch is reported as fully hard and hardware-ready.
- `run_phase3.sh`: removed all temperature, warmup, progressive flow, and hardening arguments. The default remains Adam with constant LR `0.02`, and the script now states that activations and LUT tables are binary from epoch 1.
- `tests/test_phase12_route.py`: switched Phase 3 expectations and checkpoint tests to `LUTBinaryConv2d`, removed progressive/soft-activation tests, added the always-hard flow assertion and invalid incomplete-six-input-group test, and retained the exact Bi-Real-to-bit forward/gradient test.
- `CURRENT_TECHNICAL_ROUTE.md`: replaced the obsolete progressive-flow and LR `0.001` description with the current fully hard `(s+1)/2`, identity-STE, Adam LR `0.02` constant route.
- Verification: `python -m unittest tests.test_phase12_route` passed all 21 tests in the `lut_net` environment. A GPU0 smoke test ran `BinaryComplexBitActivation -> LUTBinaryConv2d(16,32,3x3)` forward/backward; activation unique values were exactly `[0.0, 1.0]`, output shape was `(2,32,8,8)`, and input/LUT gradients were finite. `run_phase3.sh` passed `bash -n`; Python modules passed syntax compilation. Existing AMP deprecation warnings in `lut_backend.py` remain unchanged.
- Protected files: neither `complexPyTorch/lut_backend.py` nor any CUDA source was edited.

<!-- experiment-entry:phase3-binary-no-epoch-output-diagnosis-20260811 -->
## 2026-08-11 - Diagnose silent Phase 3 binary run: binary backward performance bottleneck

- Run inspected: `runs/phase3_binary_lr002/logs/train.txt`. It initialized the dataset, Phase 2 checkpoint, 36 binary LUT branches, optimizer state, and entered Epoch 1 successfully. There was no exception; the user terminated it before the first epoch completed.
- Logging behavior: the active training loop reports metrics only after a complete epoch, not per batch. With 50,000 samples and batch size 256, approximately 196 silent training batches occur before the next log line.
- Process/GPU check after termination found no active training process and idle GPUs at that moment, confirming the original job had been killed rather than continuing invisibly.
- Controlled timing isolated the issue. A complete Phase 3 model with synthetic `batch=1` completed forward in about 0.331 seconds, but backward did not complete within a 90-second timeout. A representative single `LUTBinaryConv2d(16,16,3x3)` backward also failed to finish within the timing window. During the long operation GPU0 remained at 100% SM utilization, so this is computation, not a DataLoader stall, log buffering issue, or deadlock.
- Root cause: `lut_conv_backward_ultimate_kernel` iterates over every LUT for every spatial/output block. For each LUT it clears a shared `64 x TILE_OC` gradient table, synchronizes the block repeatedly, computes six Boolean derivatives, and performs shared/global atomic accumulation. The complex operator invokes two LUT tables on both real and imaginary inputs, multiplying this already expensive backward path across 18 residual operators. The binary forward is fast; the binary backward is the bottleneck.
- This explains why the earlier floating-kernel route was much faster. The current binary CUDA kernel is suitable for hard inference forward but its training backward implementation is not practical for the full complex network.
- Recommended next implementation: keep Bi-Real activation and LUT entries hard `{0,1}` in forward, use the binary kernel only to establish the hard forward value, and route backward through the faster FP LUT kernel as a proxy/STE. Detaching the binary result prevents the slow binary backward from running. This preserves hardware-consistent forward semantics while changing only the training surrogate. A simpler equivalent may use the FP kernel directly with hard `{0,1}` inputs/tables if an equality test confirms bit-exact outputs at Boolean corners.
- No source files were changed during this diagnosis. No CUDA or `lut_backend.py` edits were made. All diagnostic benchmark processes were stopped or ended by timeout.

<!-- experiment-entry:phase3-restore-a30-grouped-binary-backend-20260811 -->
## 2026-08-11 - Correct binary slowdown diagnosis and restore A30 grouped-binary backend

- Correction to the immediately preceding diagnosis: the A30 result was valid, and the grouped binary training kernel is fast. The slow run did not use the same training backend despite both paths being described as binary LUT kernels.
- Backend identity: the real CIFAR Bi-Real reference uses `lut_conv_group -> LUT6Function -> lut_cuda_grouped_binary`. The recently restored `LUTBinaryConv2d` instead used the experimental `LUTBinaryConvFunction -> lut_conv_binary_cuda` direct-convolution backend. Existing experiment entries from 2026-08-01 had already measured that direct backend at over 30 seconds for a tiny backward and explicitly removed it from the clean route. The rollback accidentally reintroduced it.
- Architecture/build checks ruled out the initial environment hypothesis: the current machine uses RTX 4090 D GPUs (SM 8.9), and the loaded direct binary extension contains a native `sm_89` cubin. The slowdown was not caused by running an A30 `sm_80` binary on Ada.
- `complexPyTorch/complexLayers.py`: restored the validated reference dataflow without changing CUDA/backend sources. `LUTBinaryConv2d` now pads and unfolds image windows, reshapes them to `[batch, locations, groups, lut_num, 6]`, hard-binarizes LUT entries, calls the existing `LUT6Function`, sums LUT outputs, and restores NCHW output. LUT weights use the reference layout `[out_channels, lut_num, 64]`. Incomplete six-input tail groups repeat the start of the local input vector exactly as the real reference does. Bi-Real `{-1,+1}` activation followed by `(s+1)/2` remains unchanged.
- `tests/test_phase12_route.py`: updated the tail-group test for the grouped reference behavior and verified the `[out_channels, lut_num, 64]` weight layout.
- `training.py`: added optional low-frequency batch progress logging to `train_one_epoch`. The main training loop passes the logger and `--log-interval`; default interval is 25 batches. This prevents approximately three minutes of silence within an epoch.
- `run_phase3.sh`: added `LOG_INTERVAL`, default 25, and forwards it as `--log-interval`.
- `CURRENT_TECHNICAL_ROUTE.md`: documented `pad + unfold -> lut_cuda_grouped_binary`, excluded the experimental direct backend from the active route, and documented progress logging.
- Performance verification on GPU0: full Phase 3 with batch 1 completed in 0.2871 seconds forward and 0.1928 seconds backward, with gradients on all 36 LUT branches. With the actual batch size 256, forward took 0.5913 seconds, backward 0.3371 seconds, output shape was `(256,10)`, and peak allocated memory was about 4608 MiB. The prior direct backend exceeded 90 seconds in backward even at batch 1.
- Correctness verification: all 21 active-route unit tests passed; Phase 2 shared checkpoint loading retained random bimodal initialization for 12 tested LUT tables; Python/shell syntax checks passed. A three-batch CPU smoke test confirmed progress logs print at the configured interval and final batch.
- Protected files: no edits were made in this correction to `complexPyTorch/lut_backend.py` or any CUDA source. Existing pre-task modifications/build artifacts under those paths were left untouched.

<!-- experiment-entry:grouped-binary-is-hard-forward-clarification-20260811 -->
## 2026-08-11 - Clarify grouped-binary naming versus fully binary LUT forward

- `grouped` in `lut_cuda_grouped_binary` describes the CUDA tensor/work organization for batching many independent LUT6 evaluations. It does not mean grouped convolution, soft LUTs, or floating forward values. The active model uses `group_num=1`.
- Active forward value domains are fully hard: Bi-Real activation produces `{-1,+1}` and `(s+1)/2` produces exact `{0,1}` address bits; `binary_gumbel_softmax(..., hard=True)` produces exact `{0,1}` truth-table entries; the grouped binary CUDA forward thresholds each address input at `>0.5`, performs an integer table index, and returns exactly the selected binary entry for each LUT6.
- After the LUT kernel, outputs from multiple LUT6 cells are summed, so the convolution aggregate is an integer-valued multi-level result rather than one bit. This matches the real Bi-Real LUT reference and the intended LUTConv operator.
- Latent LUT logits and backward gradients remain floating point so entries can be optimized. Requiring the stored trainable parameters or gradients themselves to be binary would remove ordinary gradient-based learning; it is distinct from requiring a fully binary forward.
- No source file was changed for this clarification; only `EXPERIMENT_LOG.md` was appended.

<!-- experiment-entry:direct-bitpacked-backward-performance-explanation-20260811 -->
## 2026-08-11 - Why the direct bit-packed LUT backward is slower than grouped binary

- Bit-packing accelerates the direct forward because six input bits form one address and each 64-entry Boolean table fits in one `uint64`. It does not make training gradients binary: backward must still emit floating `grad_x` and 64 floating `grad_w` values per LUT.
- The direct kernel launches blocks over spatial positions and 32-output-channel tiles. Every block then loops over every LUT. For every LUT iteration it clears a shared `64 x 32` float gradient table, executes several block-wide synchronizations, and eventually atomically merges the shared table into the same global LUT-gradient addresses updated by many other spatial blocks.
- For a representative first-stage layer with `B=256`, `H=W=32`, `O=16`, and `L=24`, the direct launch has about 65,536 spatial blocks, each looping over 24 LUTs. This creates roughly 1.57 million shared-table initialization/synchronization rounds. Since each table has 2,048 floats, the kernel clears on the order of 3.2 billion shared float slots. The 32-output tile is also only half utilized when `O=16`.
- The grouped binary kernel launches work by `(output_channel, LUT_index, BR_chunk)`. Each block owns one output/LUT pair, uses only 64 shared LUT values and 64 shared gradient values, processes many batch/spatial samples, and aggregates locally before a much smaller number of global atomic updates. It therefore matches the reduction structure required by `grad_w` instead of the spatial lookup structure optimized for forward.
- The direct kernel's warp reduction makes `grad_x` reasonably compact, but repeated shared-table clearing, synchronization, and contended `grad_w` atomics dominate. Thus the direct design is a fast inference-forward organization paired with an unsuitable training-backward organization; the slowdown is not caused by binary arithmetic itself.
- A performant bit-packed training kernel would keep the packed forward but redesign backward around output/LUT ownership, likely separating `grad_w` and `grad_x` kernels. No source file was changed for this analysis; only `EXPERIMENT_LOG.md` was appended.


<!-- experiment-entry:grouped-direct-binary-equivalence-conditions-20260811 -->
## 2026-08-11 - Forward/backward equivalence conditions for grouped and direct binary LUT kernels

- Scope: current grouped binary LUT path (`lut_conv_group -> LUT6Function -> binary_lut`) versus the old direct bit-packed convolution path. They represent the same operator only after aligning logical connections, padding, tail handling, and LUT address mapping.
- Raw tensors cannot be copied directly: grouped weights use `[out_channels, lut_num, 64]`, while direct weights use `[lut_num, 64, out_channels]`. Grouped indexing uses `1 << (5-j)`, so input 0 is the MSB; direct `ballot_sync` puts input 0 in the LSB. Conversion requires both a layout transpose and 6-bit address reversal.
- Forward: after that conversion, every 6-bit input selects the same binary entry and the same LUT outputs are accumulated, so the mathematical outputs are exactly equal. For exact 0/1 inputs, grouped `>0.5` and direct `>0` thresholds are equivalent.
- Backward: both implement the same finite-difference surrogate. The selected entry accumulates `grad_y`; input bit k receives `grad_y * (T(addr, bit_k=1) - T(addr, bit_k=0))`, accumulated across outputs and windows. CUDA atomic/reduction order differs, so floating results should be allclose but are not guaranteed bitwise identical.
- Tail caveat: grouped repeats the beginning of the local vector when the input count is not divisible by 6. A direct implementation using floor or fixed supported counts is not equivalent in that case. The default `C=16, kernel=3x3` has `16*9=144`, divisible by 6.
- Source-code changes: none. This entry also repairs an earlier malformed append caused by shell interpretation of Markdown backticks.

<!-- experiment-entry:pair-lut4-exact-bit-phase3-20260811 -->
## 2026-08-11 - Pair-LUT4 complex convolution with exact 0/1 activation inputs

### 实验目标

重新启用“每两个复数 activation 作为一组”的 LUT-as-neuron 方案，但修复旧实验最关键的值域错误：保持当前 Bi-Real activation 和梯度不变，并在 LUT 入口显式使用 `(sign(x)+1)/2`，保证实部、虚部地址 bit 严格属于 `{0,1}`，不再把 `{-1,+1}` 直接交给 LUT。

### 当前算子定义

- 一个卷积窗口包含 `k*k*Cin` 个复数位置，按 channel/spatial 顺序 flatten。
- 每两个复数位置组成四位地址 `[x0r,x0i,x1r,x1i]`。
- 每个 pair、每个输出通道拥有两张独立 LUT4：一张输出局部实部 bit，一张输出局部虚部 bit。
- pair 数为 `ceil(k*k*Cin/2)`；实部和虚部输出分别沿 pair 维累加。
- LUT 输出采用 `{0,1}` 直接计数，因此两路聚合范围都是 `[0,pair_num]`。现有 post complex-BN 负责吸收直流偏置和尺度；算子内部没有额外复数乘法、交叉分支或 `2*count-pair_num` 变换。
- 奇数尾组重复窗口中的第一个复数位置。默认 3x3 stage 的 channel 数为 16/32/64，逻辑位置数均为偶数，不触发尾组。
- 每张 16-entry 表使用 bimodal logit 初始化；前向为 hard `0/1`，反向为 identity STE。

### CUDA 映射

本次没有修改 `complexPyTorch/lut_backend.py` 或任何 CUDA 源文件。新层复用当前 grouped binary LUT6 backend：

1. 在四个有效地址 bit 后附加两个常数 0；
2. 将每个 LUT4 entry 沿两个未使用地址位复制四份，形成 64-entry 表；
3. 调用 `LUT6Function`；
4. 对 pair 输出累加。

该映射保持 LUT4 前向严格不变，并令两个 dummy bit 的有限差分为 0。

### 文件改动摘要

- `complexPyTorch/complexLayers.py`
  - 新增 `PairLUT4ComplexConv2d`。
  - 实现复数 patch 展开、两复数配对、LUT4 CPU multilinear reference、LUT4-to-LUT6 精确嵌入、real/imag 两路累计和 hard-table STE。
  - 保留原 `LUTBinaryConv2d`、`TwoLUTComplexConv2d` 和 backend/CUDA 接口。
- `complexPyTorch/complexBinaryResNet.py`
  - 新增 `phase3_operator` 选择器，允许 `pair_lut4` 与 `shared_lut6`。
  - Phase3 默认使用 `PairLUT4ComplexConv2d`；activation 仍为现有 `BinaryComplexBitActivation`。
  - Phase1/Phase2 路由不变。
- `training.py`
  - 新增 CLI `--phase3-operator {pair_lut4,shared_lut6}`，默认 `pair_lut4`。
  - Phase2 checkpoint 初始化会跳过新 LUT4 表，只加载共享 stem/BN/projection/classifier 状态。
  - 新 LUT4 参数归入 no-weight-decay 组。
  - hard-flow 检查、nonfinite report 元数据和启动日志均识别新 operator。
- `run_phase3.sh`
  - 新增环境变量 `PHASE3_OPERATOR`，默认 `pair_lut4`。
  - 默认 workdir 改为 `runs/phase3_pair_lut4`，并将 operator 传入 training CLI。
  - 启动信息明确标记 pair-LUT4、严格 0/1 地址和 fully-hard STE。
- `tests/test_phase12_route.py`
  - 默认 Phase3 断言更新为 PairLUT4。
  - 新增 `[r0,i0,r1,i1]` 地址顺序、双输出、输入/LUT 梯度、表形状和 shared-LUT6 对照路由测试。
  - checkpoint shared-state 与 hard-flow 测试同步新 operator。
- `scripts/audit_active_route.py`
  - 路线审计改为验证 PairLUT4 默认 operator、每张表 16 entries、每层 pair 数和 0/1 activation。
  - 不再将 pair 误判为归档选项。
- `CURRENT_TECHNICAL_ROUTE.md`
  - Phase3 章节全面改写为成对 LUT4 数据流、训练定义、CUDA 嵌入、硬件含义和对照开关。
- `EXPERIMENT_LOG.md`
  - 追加本条实现、验证和运行基线，供后续结果分析复用。

### 验证结果

- Python 语法检查：通过。
- 全部单元测试：`26/26` 通过。
- active-route audit：通过。
- launcher 在 `lut_net` 环境中解析成功，显示 `--phase3-operator {pair_lut4,shared_lut6}`。
- CPU LUT4 reference 对 CUDA grouped backend，同输入、同表、同上游梯度：
  - forward max abs diff：`0.0`
  - real input grad max abs diff：`1.1920929e-6`
  - imag input grad max abs diff：`2.8610229e-6`
  - LUT grad max abs diff：`9.536743e-7`
- 完整默认 Phase3（start_filter=16, num_blocks=3）在 GPU1 单 batch 前反向：
  - 输出 shape：`(2,10)`
  - 检查到 18 个 PairLUT4 residual operator；
  - 18/18 operator 输入均严格为 `{0,1}`；
  - 18/18 LUT 表均获得 finite gradient；
  - 单 batch smoke elapsed 约 `0.461 s`。
- 仅有已存在的 PyTorch AMP deprecation warning；无功能错误。

### 推荐正式命令

从头训练：

```bash
GPU_ID=1 PHASE3_OPERATOR=pair_lut4 WORKDIR=runs/phase3_pair_lut4_bimodal_lr002 ./run_phase3.sh
```

从 Phase2 checkpoint 加载共享状态：

```bash
GPU_ID=1 PHASE3_OPERATOR=pair_lut4 CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase2.pt WORKDIR=runs/phase3_pair_lut4_from_phase2_lr002 ./run_phase3.sh
```

Matched shared-LUT6 对照仍可用 `PHASE3_OPERATOR=shared_lut6`。

<!-- experiment-entry:pair-lut4-scratch-sf11-decision-20260811 -->
## 2026-08-11 - Pair-LUT4 switches to matched-size training from scratch

### 报错原因

用户按上一条推荐命令读取 `bi_workdir/chkpts/Bestmodel_phase2.pt` 时，模型在训练开始前发生 shape mismatch。检查 checkpoint 元数据确认：

- `phase=2`
- `start_filter=11`
- `num_blocks=3`
- `spectral_pool_scheme=none`
- checkpoint epoch 126

上一条命令未设置 `START_FILTER`，而当前 `run_phase3.sh` 默认是 16，因此 stem、BN 和各 stage 通道全部出现 `11 vs 16` 的形状冲突。

将模型宽度改为 11 后，又发现该旧 checkpoint 来自更早的网络拓扑：包含 `bn_pre`，projection 使用旧的 `proj.0 Conv + proj.1 BN` 命名，而当前干净路线采用 avg-pool 后的 `proj.1 Conv + proj.2 BN`。因此该 checkpoint 不适合作为“只改变卷积算子”的严格起点。

### 最终决定

用户决定不读取 checkpoint，只要求新模型尺寸与旧模型一致，直接从头训练：

- `start_filter=11`
- `num_blocks=3`
- `phase3_operator=pair_lut4`
- 所有 backbone、BN、classifier 和 LUT 参数均重新初始化
- 显式设置 `CHECKPOINT=`，防止 shell 中遗留的 checkpoint 环境变量被使用

在排查期间曾临时加入旧 projection key 兼容映射；用户决定 from scratch 后，该临时映射已完整撤销，`training.py` 最终没有保留这一未采用方案。

### 文件改动摘要

- `training.py`
  - 临时 legacy projection 映射已添加后撤销；最终文件相对本轮开始没有保留该兼容逻辑。
  - PairLUT4 实现和 from-scratch 路由保持不变。
- `CURRENT_TECHNICAL_ROUTE.md`
  - 本轮推荐命令更新为 `start_filter=11, num_blocks=3` 的 from-scratch 实验。
  - 删除将旧 Phase2 checkpoint 作为本轮推荐起点的描述。
- `EXPERIMENT_LOG.md`
  - 追加本条故障原因、路线决策、撤销操作与验证结果。

### 验证

- 全部测试：`26/26` 通过。
- `_resolve_initial_checkpoint(args) == (None, None)`。
- GPU1 完整宽度 11 模型单 batch 前反向通过：
  - output shape `(2,10)`
  - PairLUT4 层数 `18`
  - 18/18 LUT gradient 存在且 finite
  - 总参数量 `2,029,221`
- 检查确认 `training.py` 不再包含临时 `_phase3_shared_source_keys` 映射。

### 最终运行命令

```bash
CHECKPOINT= \
GPU_ID=1 \
START_FILTER=11 \
NUM_BLOCKS=3 \
PHASE3_OPERATOR=pair_lut4 \
WORKDIR=runs/phase3_pair_lut4_scratch_sf11_lr002 \
./run_phase3.sh
```

<!-- experiment-entry:pair-lut4-riri-layout-audit-20260812 -->
## 2026-08-12 - Pair-LUT4 real/imag input layout audit

### 用户疑问

用户担心 `_pair_patches` 分别对实部和虚部执行 `F.unfold` 后，会形成“全部实部在前、全部虚部在后”的布局，使某些 LUT4 只看到实部或只看到虚部。

### 代码核对结论

当前实现不会发生该问题：

1. `real` 和 `imag` 的 shape 都是 `[B, logical_positions, locations]`。
2. `torch.stack((real, imag), dim=-1)` 立即形成 `[B, logical_positions, locations, 2]`，最后一维顺序严格为 `[real, imag]`。
3. `permute(0,2,1,3).contiguous()` 得到 `[B, locations, logical_positions, 2]`；每个复数位置的实部和虚部在连续内存中相邻。
4. 最终 `view(B, locations, pair_num, 4)` 合并两个连续复数位置，地址顺序严格为 `[x0r,x0i,x1r,x1i]`。

最小编号实测：

```text
Cin=3, k=1
real = [10,20,30]
imag = [11,21,31]
pair0 = [10,11,20,21]
pair1 = [30,31,10,11]
```

第二组展示了奇数尾组规则：最后一个复数 `[30,31]` 与窗口第一个复数 `[10,11]` 配对。

### Pair 的位置顺序

`F.unfold` 的 logical position 顺序是 channel-major，并在每个 channel 内按 kernel spatial 顺序排列。因此 pair 通常连接同一 channel 的相邻 kernel tap，也可能在 flatten 边界处连接前一 channel 的最后一个 tap 与下一 channel 的第一个 tap。但无论 pair 是否跨 channel，每个复数位置的 real/imag 始终相邻，不会拆开。

### 文件改动

没有修改模型代码；当前排列正确。仅向 `EXPERIMENT_LOG.md` 记录本次核对和编号验证。

<!-- experiment-entry:pair-lut4-channel-priority-same-spatial-20260812 -->
## 2026-08-12 - Pair-LUT4 switches to same-spatial channel-priority pairing

### 修改目标

将 PairLUT4 的复数位置展开顺序从旧的 channel-major（每个 channel 内遍历 kernel spatial）改成用户定义的 channel-priority 配对：固定一个 kernel spatial tap，在该位置内优先遍历 input channel，使每张 LUT4 的两个复数输入来自同一位置的两个相邻 channel。

### 最终数据排列

`F.unfold` 原始输出 `[B, Cin*k^2, locations]` 先恢复为：

```text
[B, Cin, kernel_area, locations]
```

随后重排为：

```text
[B, locations, kernel_area, Cin]
```

再将 real/imag stack 到最后一维：

```text
[B, locations, kernel_area, Cin, 2]
```

因此每个 LUT4 的地址顺序仍严格为：

```text
[channel c real, channel c imag,
 channel c+1 real, channel c+1 imag]
```

即 `[x0r,x0i,x1r,x1i]`，而两个复数属于同一个 kernel spatial tap。

### 奇数 channel 规则

为避免正式配置 `Cin=11` 时跨 spatial 位置配对，不再对完整的 `k^2*Cin` 序列只补一次尾组。现在对每个 spatial tap 单独补齐 channel：

- `Cin` 为偶数：直接按 `(c0,c1),(c2,c3),...` 配对；
- `Cin` 为奇数：最后一个 channel 与该 spatial tap 的第一个 channel 配对；
- 任意情况下都不跨 kernel spatial tap。

pair 数由：

```text
ceil(k^2 * Cin / 2)
```

改为：

```text
k^2 * ceil(Cin / 2)
```

因此 `Cin=11, k=3` 时，每个输出通道从旧定义的 50 个 global pair 变为 `9*6=54` 个 strictly same-spatial pair。

### 文件改动摘要

- `complexPyTorch/complexLayers.py`
  - `PairLUT4ComplexConv2d._pair_patches` 恢复 `Cin × kernel_area` 维度并执行 `spatial -> channel` 重排。
  - 新增 `pairs_per_position=ceil(Cin/2)`。
  - `pair_num` 改为 `kernel_area*pairs_per_position`。
  - 奇数 channel 在每个 spatial tap 内复制该 tap 的第一个 channel。
  - LUT 地址内部的 `[r0,i0,r1,i1]` 顺序、hard table、CUDA LUT6 嵌入和累计方式均不变。
- `tests/test_phase12_route.py`
  - 新增 `Cin=4,k=2` 唯一编号测试，验证每个 pair 是同一 spatial tap 的两个相邻 channel。
  - 新增 `Cin=3,k=2` 奇数 channel 测试，验证尾组在每个 spatial tap 内使用 `last channel + first channel`，绝不跨位置。
- `scripts/audit_active_route.py`
  - PairLUT4 表 shape 审计公式更新为 `k^2*ceil(Cin/2)`。
- `CURRENT_TECHNICAL_ROUTE.md`
  - 更新 flatten 顺序、pair 数公式和奇数 channel 尾组规则。
- `EXPERIMENT_LOG.md`
  - 追加本条实现与验证记录。
- `complexPyTorch/lut_backend.py` 和 CUDA 文件：未修改。

### 验证结果

- 全部单元测试：`28/28` 通过。
- active-route audit：通过。
- Python syntax check：通过。
- `Cin=11,k=3` CPU reference 与 CUDA grouped binary backend：
  - pair_num：`54`，符合 `9*ceil(11/2)`
  - forward max abs diff：`0.0`
  - real input grad max abs diff：`1.9073486e-6`
  - imag input grad max abs diff：`1.9073486e-6`
  - LUT grad max abs diff：`1.9073486e-6`
- 完整 `start_filter=11,num_blocks=3` GPU 模型：
  - output shape：`(2,10)`
  - PairLUT4 layers：`18`
  - 18/18 LUT gradients finite
  - 第一层 pair_num：`54`

### 兼容性说明

本次改变了 LUT table 的 pair 语义和奇数 channel 时的 pair 数，因此修改前的 PairLUT4 checkpoint 不应直接加载到新排列。当前正式实验本来就是 from scratch，不受影响。

<!-- experiment-entry:disable-intra-epoch-batch-logs-20260812 -->
## 2026-08-12 - Disable intra-epoch batch progress logs by default

### 修改目标

关闭 epoch 内部每隔若干 batch 打印的训练进度，只保留每个 epoch 结束时的汇总日志。

### 行为

`train_one_epoch` 原有判断为：

```python
if logger is not None and log_interval > 0 and ...:
```

因此 `log_interval=0` 会明确关闭 batch progress，不影响 epoch 结束后的 train/val/test loss 和 accuracy 汇总，也不影响学习率、LUT hard-flow 状态或异常诊断。

CLI 参数仍然保留；如果以后临时排查卡顿，可显式设置 `LOG_INTERVAL=25` 重新开启。

### 文件改动摘要

- `training.py`
  - `--log-interval` 默认值由 `25` 改为 `0`。
- `run_phase3.sh`
  - `LOG_INTERVAL` 默认值由 `25` 改为 `0`，并继续显式传给 training CLI。
- `tests/test_phase12_route.py`
  - 新增测试，断言默认 `training.parse_args([]).log_interval == 0`。
- `EXPERIMENT_LOG.md`
  - 追加本条变更记录。

### 验证

- 全部测试：`29/29` 通过。
- CLI 默认值实测：`log_interval 0`。
- `run_phase3.sh` shell syntax check：通过。
- 训练中不再出现 `Epoch N batch M/... train_loss...`，每个 epoch 结束汇总保持不变。

<!-- experiment-entry:triple-lut6-default-after-pair-result-20260812 -->
## 2026-08-12 - Pair-LUT4 result and Triple-LUT6 default Phase3 implementation

### 上一版 PairLUT4 完整结果

使用可复用脚本 `scripts/analyze_training_run.py` 分析
`runs/phase3_pair_lut4_scratch_sf11_lr002`：

- 状态：complete，256/256 finite epochs；
- 配置：from scratch，`start_filter=11`，`num_blocks=3`，Adam，
  constant LR `0.02`，batch 256，hard LUT + STE；
- best test：`81.10%`，epoch 116，test loss `0.595156`；
- best 点 train accuracy：`73.80%`；
- final test：`77.83%`，final train：`71.35%`；
- 无 NaN/Inf。

对比已记录的 Phase2 baseline `85.05%`，PairLUT4 低 `3.95 pp`。它比更早
analytic pair 的 `50.22%` 显著好，证明修复 `{-1,+1}` 到 `{0,1}` 地址值域
是关键；但仍未超过 Phase2，而且后期在 constant LR 0.02 下明显回落。

channel-priority PairLUT4 对 `Cin=11,k=3` 使用
`9*ceil(11/2)=54` 组。用户决定停止把该结构作为默认路线，尝试更宽的局部 neuron。

### 新方案：三个复数输入、两个 LUT6 输出

新增 `TripleLUT6ComplexConv2d`。对于每个 kernel spatial tap，沿 channel
优先顺序把三个二值复数组织为：

```text
[x0r,x0i,x1r,x1i,x2r,x2i]
```

这六个 bit 同时连接两张独立 64-entry LUT6，分别产生局部实部 bit 和虚部 bit。
所有组的两路输出分别累加，再进入现有 post complex-BN。

分组数：

```text
lut_num = k^2 * ceil(Cin / 3)
```

奇数/余数 channel 在每个 spatial tap 内独立从该 tap 的开头补足，绝不跨 spatial。
例如 `Cin=5` 使用 `(c0,c1,c2)`、`(c3,c4,c0)`。

activation 保持 `BinaryComplexBitActivation`：前向输入严格为 `{0,1}`，反向
保持 Bi-Real surrogate。LUT table 前向 hard、logit backward identity STE；
输入梯度继续使用 LUT 有限差分。

### 文件改动摘要

- `complexPyTorch/complexLayers.py`
  - 新增 `TripleLUT6ComplexConv2d`。
  - 实现 spatial 固定/channel 优先的三复数分组、每位置尾组补齐、CPU 64-state
    multilinear reference、CUDA grouped `LUT6Function` 路径、实虚两路累计。
  - 保留 `PairLUT4ComplexConv2d` 和 `TwoLUTComplexConv2d`，未删除历史方案。
- `complexPyTorch/complexBinaryResNet.py`
  - `phase3_operator` 增加 `triple_lut6` 并改为默认。
  - 保留 `pair_lut4` 和 `shared_lut6` 显式对照。
- `training.py`
  - 导入并识别 TripleLUT6 的 checkpoint LUT 参数排除、no-decay、hard-flow、
    nonfinite diagnostic 和启动统计。
  - CLI choices 更新为 `triple_lut6,pair_lut4,shared_lut6`，默认
    `triple_lut6`。
  - `log_interval=0` 保持不变，每个 epoch 只打印汇总。
- `run_phase3.sh`
  - 默认 operator 改为 `triple_lut6`。
  - 默认 workdir 改为 `runs/phase3_triple_lut6`。
  - 保持 from-scratch/checkpoint 开关和其他训练参数不变。
- `tests/test_phase12_route.py`
  - 默认 Phase3 路由断言更新为 TripleLUT6。
  - 新增六位 `ririri` 地址、实虚双输出、channel-priority、`Cin=5` 同位置尾组、
    输入/LUT 梯度测试。
  - checkpoint shared-state 与 hard-flow 测试同步 TripleLUT6。
  - PairLUT4 和 shared-LUT6 对照测试继续保留。
- `scripts/audit_active_route.py`
  - 默认路线审计更新为 `k^2*ceil(Cin/3)` 张组表、每表 64 entries、
    real/imag 两路输出。
- `CURRENT_TECHNICAL_ROUTE.md`
  - Phase3 章节更新为 TripleLUT6 数据流、上一实验结果、运行命令与资源分析。
- `EXPERIMENT_LOG.md`
  - 追加本条实验分析与实现记录。
- `complexPyTorch/lut_backend.py` 和所有 CUDA 源文件：未修改。

### 验证

- Python compile：通过。
- 全部测试：`33/33` 通过。
- active-route audit：通过。
- 默认 CLI：`phase3_operator=triple_lut6`、`log_interval=0`。
- `Cin=11,k=3` CPU/CUDA：
  - `lut_num=36=9*ceil(11/3)`
  - forward max abs diff：`0.0`
  - real input grad max abs diff：`2.8610229e-6`
  - imag input grad max abs diff：`2.1457672e-6`
  - LUT grad max abs diff：`1.9073486e-6`
- 完整 `start_filter=11,num_blocks=3` GPU smoke：
  - output `(2,10)`
  - TripleLUT6 layers `18`
  - 18/18 输入严格为 `{0,1}`
  - 18/18 LUT gradients finite
  - 第一层每个 real/imag output bank 各 36 组
  - 总 trainable parameters `5,632,997`

### 资源账

对 `Cin=11,k=3` 每个输出 channel：

- PairLUT4：54 组双输出 LUT4；若映射为一个 LUT6_2/site 每组，约 54 sites；
- TripleLUT6：36 组，每组两张单输出 LUT6，共 72 sites；
- Triple 相比 Pair 物理 LUT site 约增加 33%，但每个 neuron 同时观察三个复数；
- 软件 truth-table 参数从每组 `2*16` 增至 `2*64`，完整模型参数约
  `2.03M -> 5.63M`。

### 运行命令

```bash
CHECKPOINT= \
GPU_ID=1 \
START_FILTER=11 \
NUM_BLOCKS=3 \
PHASE3_OPERATOR=triple_lut6 \
WORKDIR=runs/phase3_triple_lut6_scratch_sf11_lr002 \
./run_phase3.sh
```

该路线必须从头开始；不要加载 PairLUT4 checkpoint。

<!-- experiment-entry:restore-pair-lut4-after-triple-lut6 -->
## 2026-08-13 - Restore PairLUT4 as default after TripleLUT6 result

### 完成实验对比

使用 `scripts/analyze_training_run.py` 重新解析两组 256-epoch from-scratch 实验：

| operator | best test acc | best epoch | final test acc | final train acc |
| --- | ---: | ---: | ---: | ---: |
| PairLUT4 | 81.10% | 116 | 77.83% | - |
| TripleLUT6 | 76.94% | 120 | 67.42% | 60.28% |

TripleLUT6 的最佳精度比 PairLUT4 低 `4.16 pp`，最终精度低 `10.41 pp`，后期回落也更明显。同时完整模型可训练参数约从 `2.03M` 增至 `5.63M`；以第一层 `Cin=11,k=3` 估算，物理 LUT site 从 54 墱至 72，约增加 33%。因此恢复 PairLUT4 为默认 Phase3 路线；TripleLUT6 保留为显式可选对照，不删除实现。

### 文件修改总结

- `complexPyTorch/complexBinaryResNet.py`：Phase3 默认 operator 恢复为 `pair_lut4`；TripleLUT6 和 shared-LUT6 映射继续保留。
- `training.py`：`--phase3-operator` 默认值恢复为 `pair_lut4`，其他 operator 仍可显式选择。
- `run_phase3.sh`：默认 operator 与默认 workdir 恢复为 PairLUT4。
- `tests/test_phase12_route.py`：默认模型、checkpoint 和结构断言恢复为 PairLUT4；TripleLUT6 的独立功能测试继续保留。
- `scripts/audit_active_route.py`：默认路线审计恢复为 PairLUT4 的两复数输入、16-entry 双输出真值表。
- `CURRENT_TECHNICAL_ROUTE.md`：恢复 PairLUT4 主路线，并补充已完成 TripleLUT6 的精度和资源结论。
- `EXPERIMENT_LOG.md`：追加本条实验分析与回退记录。
- `complexPyTorch/complexLayers.py`：本次未修改；PairLUT4 与 TripleLUT6 实现均保留。
- `complexPyTorch/lut_backend.py` 和所有 CUDA 文件：未修改。

### 验证

- 单元测试：`33/33` 通过。
- active-route audit：通过，确认 Phase3 activation 为严格 `{0,1}`，默认主卷积为 PairLUT4，真值表为 16 entries。
- `run_phase3.sh` shell syntax：沿用已验证结构，默认变量已检查为 `PHASE3_OPERATOR=pair_lut4`。

### 结论

当前已测试粒度下，增加到三个复数输入并未换来表达收益，反而使优化稳定性、最终精度和资源效率同时下降。后续继续以 PairLUT4 为基础定位精度差距，不再默认扩大 LUT 输入组。

<!-- experiment-entry:phase2-checkpoint-to-pair-lut4-analytic-init -->
## 2026-08-13 - Initialize PairLUT4 from trained Phase2 binary complex weights

### 目标与定义

将当前 PairLUT4 从随机初始化改为可由训练好的标准 Phase2 Binary Complex NN checkpoint 解析初始化。每个 LUT neuron 固定观察同一 kernel spatial tap 的两个相邻复数 input channel，四位地址顺序为 `[x0_r,x0_i,x1_r,x1_i]`。

对每个输出通道和 channel pair，读取 Phase2 的 `conv_r.weight` 与 `conv_i.weight`，使用与 Phase2 forward 相同的实虚权重符号。遍历 16 种输入 bit，将 bit 解码为 `{-1,+1}` 后计算：

```text
real = xr0*wr0 - xi0*wi0 + xr1*wr1 - xi1*wi1
imag = xi0*wr0 + xr0*wi0 + xi1*wr1 + xr1*wi1
```

分别以 `real >= 0`、`imag >= 0` 生成两个 Boolean 输出，写成 `-1/+1` LUT logits。Phase2 的 per-output-channel 正缩放因子对该符号表没有影响。奇数 `Cin` 的尾组将缺失的第二个权重置 0，因此初始真值表与两个填充地址位无关；训练后仍允许 LUT 使用这些位。

### 实现行为

- 提供 Phase2 checkpoint 且 operator 为 PairLUT4：共享的 BN、残差、stem、projection 和 classifier 参数照常加载，所有 PairLUT4 表由对应 Phase2 主卷积权重解析生成。
- 不提供 checkpoint 并使用 `--train-from-scratch`：保持原有 bimodal 随机初始化。
- TripleLUT6/shared-LUT6 没有对应的两复数解析规则，仍保持其原有随机初始化。
- 兼容旧 Binary Complex NN checkpoint 的 projection 编号：旧 `proj.[conv,bn]` 参数映射到当前 `proj.[avgpool,conv,bn]` 的 `proj.1/2`；映射后仍严格检查所有当前共享参数，缺失或 shape 不匹配会报错。

### 真实 checkpoint 检查

使用 `bi_workdir/chkpts/Bestmodel_phase2.pt`，配置 `start_filter=11,num_blocks=3`：

- 成功加载 125 个共享 tensors；
- 成功解析初始化 18 个 PairLUT4 operators；
- LUT entries 总数 2,033,856；
- logits 的唯一值严格为 `{-1,+1}`；
- hard table 中 1 的比例为 68.8279%；
- 0 个 PairLUT4 参数集保留随机初始化。

### 文件修改总结

- `complexPyTorch/complexLayers.py`：为 `PairLUT4ComplexConv2d` 增加 Phase2 二值复权重到双输出 LUT4 真值表的向量化编译方法；处理 shape、logit magnitude 和奇数 channel 尾组。
- `training.py`：Phase3 checkpoint 加载后自动解析初始化 PairLUT4；加入旧 projection 索引的受限兼容映射和初始化数量提示。
- `run_phase3.sh`：启动时区分并打印 checkpoint 解析初始化与 from-scratch bimodal 初始化。
- `tests/test_phase12_route.py`：新增真值表公式、奇数 channel 填充位独立性和旧 projection checkpoint 映射测试。
- `CURRENT_TECHNICAL_ROUTE.md`：补充完整解析公式、tie rule、尾组规则、兼容策略、真实 checkpoint 统计和运行命令。
- `EXPERIMENT_LOG.md`：追加本记录。
- `complexPyTorch/lut_backend.py` 与所有 CUDA 文件：未修改。

### 验证

- 定向初始化测试：`2/2` 通过。
- 完整单元测试：`35/35` 通过。
- active-route audit：通过。
- `run_phase3.sh` shell syntax：通过。
- 真实 Phase2 checkpoint smoke load：通过。

### 运行命令

```bash
GPU_ID=1 \
CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase2.pt \
START_FILTER=11 \
NUM_BLOCKS=3 \
PHASE3_OPERATOR=pair_lut4 \
WORKDIR=runs/phase3_pair_lut4_phase2init_sf11_lr002 \
./run_phase3.sh
```

<!-- experiment-entry:restore-pair-lut4-channel-major-spatial-pairs-20260813 -->
## 2026-08-13 - Restore PairLUT4 channel-major spatial pairing

### 修改目标

将 PairLUT4 从 2026-08-12 引入的“固定 kernel spatial tap、配对相邻 channels”恢复为最初的 channel-major 全窗口配对：一个 LUT 通常接收同一个 input channel 的两个相邻 kernel spatial positions。

### 恢复后的精确定义

`F.unfold` 的逻辑位置顺序为：

```text
channel -> kernel_row -> kernel_col
```

实部和虚部在每个复数位置内 stack，得到 `[real,imag]`，再对整个长度为 `Cin*k²` 的复数序列全局两两分组。每张 LUT4 地址保持：

```text
[x0_real, x0_imag, x1_real, x1_imag]
```

pair 数恢复为：

```text
ceil(Cin*k²/2)
```

边界行为与最初版本一致：

- 绝大多数 pair 来自同一个 channel 的两个相邻 kernel taps；
- 当 `k²` 为奇数时，channel 边界处会有一个 pair 连接前一 channel 最后一个 tap 与下一 channel 第一个 tap；
- 当 `Cin*k²` 为奇数时，全局最后一个复数位置与窗口第一个复数位置补齐。

### Phase2 checkpoint 初始化同步

`PairLUT4ComplexConv2d.initialize_from_binary_complex_weights` 同步改为使用全局 logical-position pair。每个 logical position 解析为各自的 `(channel,kernel_row,kernel_col)`，读取两个对应 Phase2 二值复权重，再遍历 16 个输入地址生成“二复数乘加 + 实虚 `>=0` 截断”的两张真值表。

全局奇数尾组运行时复制窗口第一个输入以补齐四位地址，但初始化时缺失的第二个数学权重设为 0，因此初始 LUT 对两个复制输入 bit 无关。

### 规模变化

对正式配置 `Cin=11,k=3`：

- same-spatial/channel-priority 版本：`9*ceil(11/2)=54` pairs；
- 恢复后的 channel-major 版本：`ceil(11*9/2)=50` pairs。

使用 `bi_workdir/chkpts/Bestmodel_phase2.pt` 初始化完整 `start_filter=11,num_blocks=3` 模型：

- 18 个 PairLUT4 operators 全部成功初始化；
- 全模型 LUT entries：2,022,592；
- logits 唯一值：`{-1,+1}`；
- hard 1 比例：68.7587%；
- 总 trainable parameters：2,029,221。

历史 `81.10%` PairLUT4 结果属于 same-spatial/channel-priority 布局，不能直接作为当前恢复版本的成绩；当前布局需要用 Phase2 解析初始化重新训练。

### 文件修改总结

- `complexPyTorch/complexLayers.py`
  - PairLUT4 `pair_num` 恢复为 `ceil(Cin*k²/2)`。
  - `_pair_patches` 恢复 channel-major 全局两两配对。
  - Phase2 权重解析初始化同步改为 global logical-position 权重索引。
- `tests/test_phase12_route.py`
  - 删除 same-spatial/channel-pair 断言。
  - 新增同 channel 空间配对编号测试。
  - 新增奇数 `k²` 跨 channel 边界测试。
  - 新增全局奇数尾组重复窗口首位置测试。
- `scripts/audit_active_route.py`
  - 表 shape 公式恢复为 `ceil(logical_positions/2)`。
  - 审计输出明确标记 channel-major spatial pairs。
- `CURRENT_TECHNICAL_ROUTE.md`
  - Phase3 数据流、公式、边界规则、checkpoint 初始化和资源统计全部同步。
  - 标明历史 81.10% 结果不属于当前布局。
- `EXPERIMENT_LOG.md`
  - 追加本记录。
- `training.py`、`run_phase3.sh`：本次无需修改；此前 Phase2 checkpoint 加载与启动方式继续适用。
- `complexPyTorch/lut_backend.py` 和所有 CUDA 文件：未修改。

### 验证

- PairLUT4 定向测试：`6/6` 通过。
- 完整单元测试：`36/36` 通过。
- active-route audit：通过。
- `run_phase3.sh` shell syntax：通过。
- 真实 Phase2 checkpoint smoke load：125 shared tensors、18 PairLUT4 operators 成功。
- `Cin=11,k=3,pair_num=50` CPU/CUDA 对照：
  - forward max abs diff：`0.0`
  - real input grad max abs diff：`9.54e-7`
  - imag input grad max abs diff：`7.15e-7`
  - LUT grad max abs diff：`1.43e-6`

### 运行命令

```bash
GPU_ID=1 \
CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase2.pt \
START_FILTER=11 \
NUM_BLOCKS=3 \
PHASE3_OPERATOR=pair_lut4 \
WORKDIR=runs/phase3_pair_lut4_channelmajor_phase2init_sf11_lr002 \
./run_phase3.sh
```

<!-- experiment-entry:categorical-complex-output-pair-lut4-20260813 -->
## 2026-08-13 - Optional categorical complex-output PairLUT4 parameterization

### 目标

在不改变默认 PairLUT4 行为的前提下，增加“联合复数输出”的 categorical 参数化。默认 `independent` 仍维护两张互相独立的 16-entry LUT4 logits；可选 `categorical` 为每个地址维护四个类别 logits，使实部和虚部由同一个合法复数状态联合产生。

### 数学与实现

Categorical 模式的可学习参数 shape：

```text
[Cout, pair_num, 16 addresses, 4 complex states]
```

固定码本：

```text
0 -> 00 -> -1-j
1 -> 01 -> -1+j
2 -> 10 ->  1-j
3 -> 11 ->  1+j
```

对每个地址：

```text
p = softmax(logits / 1.0)
h = one_hot(argmax(p))
selection = p + stop_gradient(h - p)
entry_bits = selection @ codebook
```

因此 forward 使用 hard argmax，生成严格 `{0,1}` 的 `[2*Cout,pair_num,16]` 双 LUT4 表；backward 通过 softmax STE 回到四分类 logits。生成双表后继续使用原有 PairLUT4 activation 查表、有限差分输入梯度、pair 累加和 complex-BN，backend/CUDA 无改动。

### 初始化

- Independent 默认模式：保持原 `[2*Cout,pair_num,16]` bimodal logits 和 hard-threshold/identity-STE。
- Categorical from-scratch：四类 logits 使用 `N(0,0.01)` 打破 argmax 平局。
- Phase2 checkpoint：先按两个二值复权重生成原解析双表，再将 `(real_bit,imag_bit)` 编码为类别 `2*real+imag`；目标类别 logit 设为 0.25，其他三类为 0。
- 同一 Phase2 权重下，independent 与 categorical 的初始 hard 双表逐 entry 完全一致。

### 开关

CLI：

```text
--pair-lut-parameterization {independent,categorical}
```

Shell：

```text
PAIR_LUT_PARAMETERIZATION=independent|categorical
```

默认值始终是 `independent`。

### 文件修改总结

- `complexPyTorch/complexLayers.py`
  - `PairLUT4ComplexConv2d` 增加 `parameterization` 与固定四状态码本。
  - Independent 参数 shape 与行为保持不变。
  - Categorical 增加 hard argmax + softmax STE 的双 LUT 表生成。
  - Phase2 解析初始化按参数化模式写入独立 logits 或四分类 logits。
  - `forward` 统一消费 `materialize_table()`，后续查表路径不变。
  - `extra_repr` 显示参数化模式。
- `complexPyTorch/complexBinaryResNet.py`
  - 网络和残差块增加 `pair_lut_parameterization`，默认 independent。
  - 该参数只传给 PairLUT4，不影响 TripleLUT6/shared-LUT6。
- `training.py`
  - 增加 `--pair-lut-parameterization` CLI、模型传参和启动日志。
  - checkpoint 提示从 bimodal 泛化为 random，兼容两种模式。
- `run_phase3.sh`
  - 增加 `PAIR_LUT_PARAMETERIZATION` 环境变量，默认 independent。
  - 启动提示区分独立阈值 STE、categorical argmax/softmax STE 和各自随机初始化。
- `tests/test_phase12_route.py`
  - 测试默认 independent shape 不变。
  - 测试四类到 `00/01/10/11` 码本映射、hard table 二值性、logits/input 梯度。
  - 测试 Phase2 初始化后两种参数化 hard table 完全一致。
- `scripts/audit_active_route.py`
  - 新增默认参数化必须为 independent 的审计。
- `CURRENT_TECHNICAL_ROUTE.md`
  - 增加两种参数化、数学定义、初始化和运行命令。
- `EXPERIMENT_LOG.md`
  - 追加本记录。
- `complexPyTorch/lut_backend.py` 与所有 CUDA 文件：未修改。

### 验证

- Categorical 定向测试：`4/4` 通过。
- 完整测试：`40/40` 通过。
- active-route audit：通过。
- Python compile 与 `run_phase3.sh` shell syntax：通过。
- `Cin=11,k=3,pair_num=50` categorical CPU/CUDA：
  - forward max abs diff：`0.0`
  - real/imag input grad max abs diff：`1.43e-6`
  - categorical logit grad max abs diff：`7.15e-7`
  - materialized table values：严格 `{0,1}`。
- 真实 `Bestmodel_phase2.pt` 完整 categorical GPU smoke：
  - 18 个 PairLUT4 全部解析初始化；
  - 第一层 categorical shape `(11,50,16,4)`；
  - 18/18 层 logits gradients finite；
  - output shape `(2,10)`；
  - trainable parameters `4,051,813`，仅增加训练期 logits，FPGA 双 LUT4 资源不变。
- `git diff --check` 仍报告 `complexLayers.py` 旧区域原有 trailing whitespace；本次新增区域没有引入该问题，未清理无关历史格式。

### Categorical 运行命令

```bash
GPU_ID=1 \
CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase2.pt \
START_FILTER=11 \
NUM_BLOCKS=3 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical \
WORKDIR=runs/phase3_pair_lut4_categorical_phase2init_sf11_lr002 \
./run_phase3.sh
```

<!-- experiment-entry:generalized-categorical-complex-lut-k4-k6-20260814 -->
## 2026-08-14 - Generalize categorical complex LUT from k=4 to k=4/6

### 新基线结果

先使用可复用脚本 `scripts/analyze_training_run.py` 解析完成的 categorical LUT4 实验：

```text
run: runs/phase3_pair_lut4_categorical_phase2init_sf11_lr002
epochs: 256/256, all finite
optimizer: Adam
initial LR: 0.02
schedule: multistep
best test: 82.53% at epoch 248
final test: 81.80%
```

该结果高于此前 PairLUT4 版本，说明联合四状态输出参数化在调整 LR schedule 后是有效路线。本次以它作为三复数 LUT6 的直接基线。

### 修改目标

将当前写死为“两个复数输入、16 个地址”的 categorical LUT4 泛化为由 LUT 输入 bit 数 `k` 配置：

```text
k=4: 2 complex inputs/group, 16 addresses, categorical logits 4*16/address bank
k=6: 3 complex inputs/group, 64 addresses, categorical logits 4*64/address bank
```

这里更精确的完整 tensor shape 为：

```text
categorical: [Cout, group_num, 2^k, 4]
independent: [2*Cout, group_num, 2^k]
group_num = ceil(Cin*kernel^2 / (k/2))
```

默认仍为 `k=4 + independent`，所以不带新参数的现有命令和模型结构不变。

### 数据流

- `F.unfold` 继续按 `channel -> kernel_row -> kernel_col` 展开复数位置。
- `k=4` 每两个连续复数一组，地址为 `riri`。
- `k=6` 每三个连续复数一组，地址为 `ririri`。
- 尾组不足时，从窗口开头复制所需复数位置补齐。
- Categorical 每个地址的四类仍编码 `00/01/10/11`，forward hard argmax，backward softmax STE，`tau=1.0`。
- 生成的两个 hard table 分别作为实部、虚部输出并沿 group 维累加。

### Phase2 初始化

初始化函数同步泛化：对于每组 `n=k/2` 个二值复权重，遍历 `2^k` 个输入状态，计算 `sum(x_t*w_t)` 后以实部/虚部 `>=0` 产生目标类别。尾组复制输入对应的数学权重设为 0。该解析 operation 只作为初始化，不宣称等价于 Phase2 整层累加。

### CUDA 与硬件

- `k=4` 保持原路径：添加两个 dummy 0 bits，并把 16-entry 表扩展为 64-entry 后调用 LUT6 backend。
- `k=6` 直接把六位地址与 64-entry 双表交给现有 LUT6 backend。
- 两种模式最终都是两张硬 LUT；categorical 的四分类 logits 只存在于训练，不增加 FPGA 推理资源。
- `complexPyTorch/lut_backend.py` 与所有 CUDA 文件均未修改。

### 文件修改总结

- `complexPyTorch/complexLayers.py`
  - 将 `PairLUT4ComplexConv2d` 内部泛化为 `lut_inputs in {4,6}`；类名保留以兼容当前路由和 checkpoint 识别。
  - 新增 `complex_inputs_per_lut`、`table_size=2^k`、`group_num`；旧 `pair_num` 保留为兼容别名。
  - 输入分组、state bits、independent/categorical 参数 shape、Phase2 初始化、CPU reference 和 CUDA dispatch 全部动态化。
  - `_pair_patches` 保留为 `_group_patches` 的兼容入口。
- `complexPyTorch/complexBinaryResNet.py`
  - 网络和残差块新增 `pair_lut_inputs=4`，并只传给当前 grouped complex LUT operator。
  - `_make_stage` 使用统一 kwargs，保证 18 个 residual operators 配置一致。
- `training.py`
  - 新增 `--pair-lut-inputs {4,6}`，默认 4；传入模型并写入启动配置日志。
- `run_phase3.sh`
  - 新增 `LUT_INPUTS=4|6`，默认 4；传给 CLI 并显示每组复数输入数量。
  - 保留用户当前 `LR=0.02`、`SCHEDULE=multistep` 默认设置。
- `tests/test_phase12_route.py`
  - 测试默认 k=4 不变。
  - 新增 k=6 categorical shape、三复数 channel-major 分组、尾组补齐、64-entry hard table、Phase2 初始化及输入/logit 梯度测试。
- `scripts/audit_active_route.py`
  - shape 审计泛化为 `group_num` 和 `2^k`；锁定默认 k=4 independent。
- `CURRENT_TECHNICAL_ROUTE.md`
  - Phase3 与 LUT 参数化章节更新为统一 k=4/6 方案，并记录 82.53% LUT4 基线和 LUT6 命令。
- `EXPERIMENT_LOG.md`
  - 追加本记录。

### 验证

- k=4/k=6 定向测试：`5/5` 通过。
- 完整单元测试：`44/44` 通过。
- active-route audit、Python compile、shell syntax、CLI 解析：通过。
- k=6 `Cin=11,kernel=3` categorical CPU/CUDA：
  - groups `33`
  - logits shape `(3,33,64,4)`（测试层 Cout=3）
  - forward max abs diff `0.0`
  - real input grad max abs diff `9.54e-7`
  - imag input grad max abs diff `7.15e-7`
  - categorical logit grad max abs diff `2.38e-7`
  - materialized table 严格 `{0,1}`。
- 真实 Phase2 checkpoint 完整 k=6 categorical GPU smoke：
  - 125 shared tensors loaded
  - 18 grouped complex LUT operators initialized
  - 第一层 shape `(11,33,64,4)`
  - 18/18 categorical gradients finite
  - output `(2,10)`
  - trainable parameters `10,786,277`，增加部分仅为训练期 logits。

### LUT6 运行命令

```bash
GPU_ID=1 \
CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase2.pt \
START_FILTER=11 \
NUM_BLOCKS=3 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical \
LUT_INPUTS=6 \
LR=0.02 \
SCHEDULE=multistep \
WORKDIR=runs/phase3_lut6_categorical_phase2init_sf11_lr002 \
./run_phase3.sh
```


<!-- experiment-entry:parallel-dominance-lut6-random-init-20260814 -->
## 2026-08-14 - 两复数并行 dominance-bit LUT6 随机初始化分支

### 目标与设计决定

- 放弃从当前 LUT4 checkpoint 复制 16-entry 真值表；新的 64-entry categorical LUT6 logits 使用 `N(0,0.01)` 独立随机初始化。
- 该分支仍只组合两个复数位置，不使用此前“三个复数压缩成一个复数”的 LUT6 地址。
- 每个复数从同一个量化前激活并行生成 `sr=1[r>0]`、`si=1[i>0]`、`p=1[|r|>|i|]`。
- 两个复数的 LUT6 地址固定为 `[sr0,si0,p0,sr1,si1,p1]`。
- categorical 参数 shape 为 `[Cout,ceil(Cin*k*k/2),64,4]`，四类对应输出 `00/01/10/11`。
- LUT 表 forward 使用 hard argmax，backward 保留当前 softmax STE；最终实部和虚部仍是严格 `{0,1}`。

### dominance 梯度

`p` 前向使用严格比较器，tie (`|r|==|i|`) 取 0。反向代理为：

```text
delta = |real| - |imag|
p_soft = clamp(0.5 + delta/(2*margin), 0, 1)
p = p_soft + stop_gradient(p_hard - p_soft)
```

因此 LUT 输出对第 3/6 个地址 bit 的有限差分梯度能够继续经过 `abs(real)-abs(imag)` 回到量化前激活。默认 `margin=1.0`，可通过 `DOMINANCE_STE_MARGIN` 配置。

### checkpoint 规则

- 不传 `CHECKPOINT`：整个模型随机训练。
- 传 Phase2 checkpoint：共享 stem、BN、shortcut、classifier 参数正常加载，但所有 dominance LUT6 logits 保持构造时随机值。
- dominance 模式禁止调用 Phase2 复权重解析编译函数，从代码层面避免意外退化为忽略 `p` 的 LUT4 扩展。
- 不读取也不需要任何 LUT4 checkpoint。

### 文件修改总结

- `complexPyTorch/complexLayers.py`
  - `PairLUT4ComplexConv2d` 新增 `activation_encoding={standard,dominance}` 与 `dominance_ste_margin`。
  - dominance 模式强制 `lut_inputs=6 + categorical`，但 `complex_inputs_per_lut=2`。
  - 新增硬比较/软反向 dominance bit，并按 `[sr0,si0,p0,sr1,si1,p1]` 分组。
  - forward 接收可选 `dominance_source`；旧 standard LUT4/三复数 LUT6 行为不变。
  - dominance LUT6 显式拒绝 Phase2 权重解析初始化。
- `complexPyTorch/complexBinaryResNet.py`
  - 残差块和顶层模型新增编码与 STE margin 参数。
  - dominance 模式把量化前激活并行传给 LUT 层；主 activation 仍沿用现有 Bi-Real `{0,1}` 编码。
  - stage 构造统一传播新参数；默认 standard 路线不变。
- `training.py`
  - 新增 `--pair-lut-encoding` 和 `--dominance-ste-margin`。
  - Phase2 shared loader 跳过 dominance LUT6 编译并报告随机 LUT 数量。
  - 模型构造和启动日志记录实际 encoding。
- `run_phase3.sh`
  - 新增 `PAIR_LUT_ENCODING`、`DOMINANCE_STE_MARGIN` 环境变量。
  - dominance 启动日志明确两复数六位地址以及 checkpoint 只加载共享参数。
- `tests/test_phase12_route.py`
  - 新增 dominance LUT6 shape/group 数测试。
  - 新增六位地址顺序及 dominance 梯度回到原始实虚部的测试。
  - 新增 Phase2 checkpoint 不覆盖随机 LUT6 logits 的测试。
- `CURRENT_TECHNICAL_ROUTE.md`
  - 追加并行 dominance-bit LUT6 路线、梯度定义、随机初始化规则和运行参数。
- `EXPERIMENT_LOG.md`
  - 追加本实验设计、实现和验证记录。

### 保护边界

`complexPyTorch/lut_backend.py` 与所有 CUDA 源文件均未修改。新模式直接复用现有 `LUT6Function`。

### 验证

- Python syntax 与 `run_phase3.sh` shell syntax：通过。
- dominance 定向测试加旧 LUT4/LUT6 初始化测试：`5/5` 通过。
- 完整 `tests.test_phase12_route`：`45/45` 通过。
- GPU1 CUDA smoke：hard output 正常；LUT logits、raw real、raw imag 梯度均有限且非零。
- smoke gradient absolute sums：LUT `0.9990`，raw real `1.0200`，raw imag `1.0600`。
- 完整小模型 GPU1 forward/backward：output shape `(2,10)`，6/6 dominance LUT6
  的 logits gradient finite 且非零，stem gradient sum `34.8355`。

### 推荐运行命令

```bash
GPU_ID=1 \
CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase2.pt \
START_FILTER=11 \
NUM_BLOCKS=3 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical \
LUT_INPUTS=6 \
PAIR_LUT_ENCODING=dominance \
DOMINANCE_STE_MARGIN=1.0 \
LR=0.02 \
SCHEDULE=multistep \
WORKDIR=runs/phase3_lut6_dominance_randomlut_phase2shared_sf11_lr002 \
./run_phase3.sh
```


<!-- experiment-entry:dominance-lut6-epoch100-analysis-20260814 -->
## 2026-08-14 - dominance LUT6 epoch 100 阶段性分析

### 日志预处理

先调用可复用的 `scripts/analyze_training_run.py` 解析 dominance、categorical LUT4
基线和旧三复数 LUT6；随后新增并调用
`scripts/analyze_dominance_lut_checkpoint.py` 审计 64-entry categorical table
对 `p0/p1` 的 hard-category 敏感度。

### 当前结果

- dominance LUT6 已记录 100 epochs，所有 loss/gradient 指标 finite。
- epoch 96 best test accuracy：`75.37%`；epoch 100：`75.17%`。
- epoch 100 train accuracy：`67.64%`，train loss：`1.2459`。
- LR 在 epoch 91 从 `0.02` 降至 `0.002`；test accuracy 从 epoch 90 的
  `71.68%` 恢复到约 `75%`，但随后再次平台化。
- 同 epoch 98，dominance 为 `74.79%`，categorical LUT4 基线为 `79.81%`；
  train accuracy 分别为 `67.49%` 和 `73.86%`。
- 完整 categorical LUT4 best 为 `82.53%`；当前 dominance best 落后 `7.16 pp`。
- 旧三复数 categorical LUT6 在 epoch 66 best `74.46%`；dominance 比它略高，
  但仍明显低于 LUT4。

### LUT hard-table 审计

best epoch 96 checkpoint：

- `p0` category-change fraction：`69.866%`。
- `p1` category-change fraction：`69.931%`。
- 单个实部/虚部输出 bit change fraction 约 `45.5%`。
- 四个输出类别在各层几乎均匀占据约 `25%`。

epoch 100 last checkpoint：

- `p0` category-change fraction：`69.856%`。
- `p1` category-change fraction：`69.902%`。
- 与 epoch 96 几乎完全相同，表结构已没有明显整理趋势。

随机四分类真值表的理论 category-change fraction 为 `75%`，单输出 bit change
fraction 为 `50%`。当前 table 虽从随机值略微形成了不变性，但仍保留大部分随机
`p` 依赖。dominance 并非没有进入 LUT；问题是它以接近随机的方式强烈控制输出。

### 原因判断

Phase2 checkpoint 在该模式下只恢复 stem、BN、shortcut 和 classifier。18 个主卷积
全部被随机 64-entry LUT6 替代，因此 Phase2 已学到的主干 operation 实际全部丢失。
这更接近“从随机局部 operator 重新训练网络”，不是在稳定 Phase2/LUT4 operation
上测试额外 dominance 信息。训练精度同步落后证明主要是优化/表达组织失败，而不是
单纯过拟合。

### 决策与下一步

- 当前实验继续到 256 epoch 的价值较低，建议停止并保留 epoch 96 best checkpoint。
- 不据此否定 dominance bit 本身；本实验否定的是“所有 LUT6 entry 完全随机初始化”
  这一训练起点。
- 下一实验应从 Phase2 复权重解析生成 16-entry C4 operation，再沿 `p0/p1` 复制为
  64 entries，保证 hard forward 初始等价于稳定基线；这不需要 LUT4 checkpoint。
- 为避免复制后 `p` 初始梯度为零，可只在 backward 增加 detached soft-table
  finite-difference surrogate，使 hard forward 完全不变但 `p` 从第一个 epoch
  就能向量化前激活传梯度。两个 p slice 的 logits 仍独立更新，之后允许自然分化。

### 文件修改总结

- `scripts/analyze_dominance_lut_checkpoint.py`
  - 新增 categorical LUT6 checkpoint 审计，统计类别占用、logit margin 以及
    `p0/p1` 对类别和两个输出 bit 的敏感率。
  - 新增 `--summary-only`，便于后续运行快速输出聚合结果。
- `EXPERIMENT_LOG.md`
  - 追加本次 epoch 100 对比、checkpoint 审计、原因判断和下一步建议。
- 本次分析未修改模型、training flow、LUT backend 或 CUDA 源码。


<!-- experiment-entry:lut4-to-dominance-lut6-warm-start-20260814 -->
## 2026-08-14 - categorical LUT4 → dominance LUT6 strict warm start

### 修改动机

完全随机 dominance LUT6 在 epoch 100 仅达到 `75.37%` best，且 hard table 的
`p0/p1` category-change fraction 仍约 `69.9%`，接近随机四分类表的 `75%`。
因此将主实验起点改为当前最佳 categorical LUT4 Phase3 checkpoint，使新模型
初始函数严格等价于已达到 `82.53%` 的 LUT4。

### 地址映射

旧地址 bit 顺序：

```text
[r, i, r', i']
```

新地址 bit 顺序：

```text
[r, i, p, r', i', p']
```

对每个新 6-bit state，旧地址计算为：

```text
old_index = 8*r + 4*i + 2*r' + i'
```

随后复制旧 tensor `old_logits[..., old_index, :]`。复制的是完整四类 logits，
不是只复制 argmax class，因此 hard LUT 输出和 softmax backward proxy 均继承。

需要注意：由于两个 dominance bit 插在地址中间，旧 address `0000` 对应的新
整数地址为 `[0,1,8,9]`，不是连续的 `[0,1,2,3]`。实现按 bit 语义 index-select，
不使用错误的 `repeat_interleave(4)`。

### checkpoint 行为

- dominance 模式读取 Phase3 checkpoint 时，要求每层 source weight shape 为
  `[Cout,group_num,16,4]`，随后扩展到 `[Cout,group_num,64,4]`。
- stem、BN、shortcut、classifier 等 topology-compatible tensors 同时加载。
- dominance 模式读取 Phase2 checkpoint 时仍保留此前行为：共享参数加载，LUT6 随机。
- standard LUT4 读取 Phase2 checkpoint 的解析初始化行为不变。

### 文件修改总结

- `training.py`
  - 新增 `_load_dominance_lut6_warm_start()`。
  - Phase3 初始化入口自动识别 Phase3 categorical LUT4 checkpoint，并执行
    语义地址扩展；Phase2 loader 继续兼容。
  - checkpoint help 更新为同时说明 Phase2 shared init 与 LUT4 warm start。
- `run_phase3.sh`
  - dominance checkpoint 日志更新：Phase3 checkpoint 执行 LUT4 扩展，
    Phase2 checkpoint 保留随机 LUT。
- `tests/test_phase12_route.py`
  - 新增完整 logits 映射测试，锁定旧 `0000 -> 新 [0,1,8,9]`。
  - 新增 LUT4 与 warm-start LUT6 整模型初始输出逐元素等价测试。
- `CURRENT_TECHNICAL_ROUTE.md`
  - 将 dominance 主初始化从随机改为 categorical LUT4 strict warm start。
- `EXPERIMENT_LOG.md`
  - 追加本实现、地址语义、验证和运行命令。

### 验证

- 定向 warm-start/Phase2 compatibility/dominance gradient 测试：`3/3` 通过。
- 完整 active-route tests：`46/46` 通过。
- 小模型 warm start 后 LUT4 与 LUT6 logits 映射逐元素相同，整模型输出
  `allclose(atol=1e-6)`。
- 真实 82.53% checkpoint：加载 `125` 个共享 tensors，扩展 `18/18` LUT layers，
  `max_logit_diff=0.0`。
- 真实地址审计：旧 address 0 映射到新 `[0,1,8,9]`。
- `complexPyTorch/lut_backend.py` 与 CUDA 文件均未修改。

### 运行命令

```bash
GPU_ID=1 \
CHECKPOINT=runs/phase3_pair_lut4_categorical_phase2init_sf11_lr002/chkpts/Bestmodel_phase3.pt \
START_FILTER=11 \
NUM_BLOCKS=3 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical \
LUT_INPUTS=6 \
PAIR_LUT_ENCODING=dominance \
DOMINANCE_STE_MARGIN=1.0 \
LR=0.02 \
SCHEDULE=multistep \
WORKDIR=runs/phase3_lut6_dominance_lut4warm_sf11_lr002 \
./run_phase3.sh
```
<!-- experiment-entry:dominance-stop-gradient-20260814 -->
## 2026-08-14 - dominance 第三 bit 默认截断反向梯度

### 决策

当前 dominance LUT6 的地址仍为 `[sr0,si0,p0,sr1,si1,p1]`，其中
`p=1[|real|>|imag|]`，forward、tie 规则和 LUT4 warm start 均不改变。
本次只切断 LUT 输出经由 `p` 回到量化前 activation 的代理梯度。sign real/imag
的 Bi-Real 梯度、categorical LUT logits 的 softmax STE 梯度、BN 和其他网络参数
梯度均保持原样。

新增 `dominance_grad_mode={stop,ste}`：

- `stop`（默认）：`p` 是 detached hard comparator，不向 `real/imag` 回传梯度。
- `ste`：保留旧 clipped-linear STE，用于精确复现此前 dominance 实验。
- `DOMINANCE_STE_MARGIN` 只在 `ste` 模式生效。

### 历史第三 bit 路线核对

- magnitude bit：`1[|x|>=learnable_threshold]`，曾使用 sigmoid STE，并让梯度同时到
  magnitude threshold 和原 activation；早期版本曾出现第五 bit 无有效梯度。
- dominance bit：`1[|real|>|imag|]`，即当前并行比较器；尝试过 hard comparator、
  clipped-linear STE 和 sigmoid/temperature soft dominance。
- C8 phase bit：仍以 sign real、sign imag、dominance 三 bit 编码，但进一步解码为
  8 相位码本；尝试过 roots（轴上 0/45/90 度系列）与 octants（扇区中心）码本，
  以及 bireal_ste、semantic_ste、semantic_phase_ste。
- threshold selector：额外 bit 曾用于在两个局部输出 threshold 之间选择，而不是
  直接改变 activation 码本；包括可学习 threshold、固定 >0/>=0 等变体。
- 两 bit activation/LUT6 是另一条增加信息带宽的路线，并非单独第三 bit。

### 文件修改总结

- `complexPyTorch/complexLayers.py`：为 dominance 增加 stop/ste 模式，默认 hard bit detach。
- `complexPyTorch/complexBinaryResNet.py`：模型和残差块传播并保存 gradient mode。
- `training.py`：增加 CLI 参数、模型传参和启动日志。
- `run_phase3.sh`：增加 `DOMINANCE_GRAD_MODE`，默认 stop，并仅在 ste 时打印 margin。
- `tests/test_phase12_route.py`：旧梯度测试显式使用 ste；新增默认 stop 回归测试。
- `CURRENT_TECHNICAL_ROUTE.md`：更新当前默认梯度语义和运行配置。
- `EXPERIMENT_LOG.md`：记录本次决策、历史路线及验证结果。

### 验证

- Python syntax 与 shell syntax 检查通过。
- dominance stop、legacy ste、LUT4-to-LUT6 warm start 三个定向测试通过。
- `tests.test_phase12_route` 完整测试进程成功退出。

<!-- experiment-entry:dominance-stopgrad-completed-analysis-20260819 -->
## 2026-08-19 - dominance stop-gradient 完整结果

使用现有 `scripts/analyze_training_run.py` 和
`scripts/analyze_dominance_lut_checkpoint.py` 重新分析最新完整运行。

### 对比结果

| 实验 | best test | best epoch | final test |
| --- | ---: | ---: | ---: |
| categorical LUT4 基线 | 82.53% | 248 | 81.80% |
| LUT6 dominance warm-start + legacy STE | 79.28% | 2 | run stopped at epoch 31 |
| LUT6 dominance warm-start + stop-gradient（第一组） | 79.95% | 1 | 78.40% |
| LUT6 dominance warm-start + stop-gradient（最新复跑） | 79.75% | 3 | 77.09% |

最新复跑共 256 epochs，所有指标 finite。stop-gradient 相比旧 STE 略有改善，
但没有超过 LUT4 起点，也没有在后期恢复。

### LUT slice 分化

- 最新 stop-gradient best checkpoint 中，改变 `p0` 时 categorical output 改变
  11.23%，改变 `p1` 时改变 11.22%；单个 real/imag bit 改变约 6.23%。
- 第一组 stop-gradient best checkpoint 的 categorical 改变约 6.44%，单输出 bit
  改变约 3.56%。
- 这说明 dominance slices 并非完全相同，第三 bit 确实改变了硬 LUT；但这些早期
  分化没有转化成准确率收益。

### 判断

stop-gradient 去除了额外 activation surrogate gradient 的干扰，但不是主要瓶颈。
LUT4 的 16-entry table 扩展成 LUT6 的 64-entry table 后，每个旧地址被
`(p0,p1)` 拆成四个独立 slice。每个 slice 的命中样本量大约降为原来的四分之一，
参数量增加四倍，hard argmax + softmax STE 的更新因而更稀疏、更噪声化。
`LR=0.02` 使复制得到的四个等价 slice 在最初几个 epoch 就快速分裂。

LUT4 在 epoch 80 附近也曾降到约 77.8%，但学习率下降后恢复到 81.7%以上；
LUT6 后期只恢复到 77–78%，说明问题不只是初始学习率，而是无约束四路表分裂
形成了更差的优化空间。

### 后续优先方案

不再优先更换第三 bit。建议使用共享基表加零均值 residual：

[
L_6(a,p_0,p_1)=L_4(a)+alpha(D(a,p_0,p_1)-mean_p D(a,p)).
]

初始化 `D=0, alpha=0`，保持与 LUT4 严格等价；LUT4 base 固定或使用低学习率，
dominance residual 使用较小学习率，并逐渐增大 `alpha`。训练完成后仍可物化为
普通 64-entry LUT6，不增加最终硬件资源。

<!-- experiment-entry:dominance-shared-base-residual-implementation-20260819 -->
## 2026-08-19 - 共享 LUT4 基表加零均值 dominance residual

### 动机

完整 stop-gradient 实验仍低于 categorical LUT4。主要问题由 activation 梯度转为
LUT4 的每个地址被两个 dominance bit 拆成四个独立、低命中率的 LUT6 slice。
同时旧实验从成熟 LUT4 checkpoint 重新以 Adam LR=0.02 启动，进一步放大了漂移。

### 新参数化

新增 `pair_lut_parameterization=categorical_residual`：

[
Z_6(a,p)=B_4(a)+alpha(D_6(a,p)-mean_p D_6(a,p)).
]

- `dominance_base`：shape `[Cout,group,16,4]`，从 LUT4 checkpoint 读取，
  作为冻结 buffer。
- `weight`：shape `[Cout,group,64,4]`，表示可学习 residual，初始化严格为零。
- `dominance_assignment`：缓存 64 个地址到 16 个旧地址的固定归属矩阵。
- 每个旧地址下四个 `(p0,p1)` residual 强制零均值，避免 residual 整体覆盖基表。
- effective logits 仍经过 hard argmax forward + softmax STE backward，最终仍是普通
  64-entry LUT6，不增加部署硬件。
- dominance bit 保持 hard forward + stop-gradient。

### 训练控制

- `DOMINANCE_RESIDUAL_LR`：residual 独立 LR，默认 0.002。
- `DOMINANCE_RESIDUAL_ALPHA_START`：默认 0.1。
- `DOMINANCE_RESIDUAL_ALPHA_END`：默认 1.0。
- `DOMINANCE_RESIDUAL_RAMP_EPOCHS`：默认 120。
- optimizer group 使用相对 `lr_scale`，因此全局 schedule 衰减时 residual LR 同比例衰减。
- checkpoint metrics 保存当前 residual alpha。
- residual checkpoint 可以直接重新加载；旧 categorical LUT4 checkpoint 自动加载到
  frozen base，并把 residual 清零。

### 文件修改总结

- `complexPyTorch/complexLayers.py`
  - 新增 `categorical_residual` 参数化、冻结 LUT4 base、零均值 residual、
    alpha buffer 和 effective-logit 构造。
  - 缓存 dominance address assignment，避免每次 forward 重建 one-hot。
- `complexPyTorch/complexBinaryResNet.py`
  - 模型验证允许 dominance LUT6 使用 `categorical_residual`；旧模式不变。
- `training.py`
  - warm-start loader 将旧 LUT4 logits 装入 base 并清零 residual。
  - 新增 residual checkpoint reload、独立 optimizer group/LR scale、alpha ramp、
    参数校验、日志和 checkpoint metric。
- `run_phase3.sh`
  - 新增 residual LR 与 alpha 环境变量并传入 CLI；修正新 categorical 模式的提示。
- `tests/test_phase12_route.py`
  - 新增严格 LUT4 等价 warm start、零均值约束、alpha schedule 和独立 LR 测试。
- `scripts/analyze_dominance_lut_checkpoint.py`
  - 对 residual checkpoint 重建实际 effective logits 后再分析 sensitivity；
    继续兼容旧 unrestricted LUT6 checkpoint。
- `CURRENT_TECHNICAL_ROUTE.md`
  - 新增共享基表 residual 路线、公式、硬件语义和推荐命令。
- `EXPERIMENT_LOG.md`
  - 记录本次实现、验证和运行配置。
- 未修改 `complexPyTorch/lut_backend.py` 及任何 CUDA kernel。

### 验证

- Python 及 shell syntax 检查通过。
- 活动路线完整测试 `50/50` 通过。
- warm-start 模型初始输出与 LUT4 模型在 `atol=1e-6` 下严格一致。
- residual analyzer smoke test 初始 `p0/p1 category sensitivity=0`，符合零 residual。
- 旧 unrestricted LUT6 checkpoint 的 analyzer 输出保持原数值。

### 推荐命令

```bash
GPU_ID=0 \
CHECKPOINT=runs/phase3_pair_lut4_categorical_phase2init_sf11_lr002/chkpts/Bestmodel_phase3.pt \
START_FILTER=11 NUM_BLOCKS=3 NUM_EPOCHS=200 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical_residual \
LUT_INPUTS=6 PAIR_LUT_ENCODING=dominance \
DOMINANCE_GRAD_MODE=stop \
LR=0.0002 SCHEDULE=constant \
DOMINANCE_RESIDUAL_LR=0.002 \
DOMINANCE_RESIDUAL_ALPHA_START=0.1 \
DOMINANCE_RESIDUAL_ALPHA_END=1.0 \
DOMINANCE_RESIDUAL_RAMP_EPOCHS=120 \
WORKDIR=runs/phase3_lut6_dominance_sharedres_lr2e4_reslr2e3 \
./run_phase3.sh
```
<!-- experiment-entry:dominance-shared-residual-success-20260819 -->
## 2026-08-19 - dominance shared-residual 达到 84.28%

使用 `scripts/analyze_training_run.py` 和
`scripts/analyze_dominance_lut_checkpoint.py` 分析完整运行：

`runs/phase3_lut6_dominance_sharedres_lr2e4_reslr2e3`

### 最终结果

- 训练完成：200/200 epochs，所有指标 finite。
- best test accuracy：84.28%，epoch 194。
- final test accuracy：84.10%，仅比 best 低 0.18 pp，后期稳定。
- LUT4 categorical 起点：82.53%，提升 1.75 pp。
- 旧 unrestricted dominance LUT6 best：79.75%，提升 4.53 pp。
- epoch 1：81.97%。
- epoch 60：82.63%。
- epoch 119：83.47%。
- epoch 120 alpha 到 1.0：83.09%。
- epoch 194：84.28%。

### 第三 bit 使用情况

best checkpoint 中：

- 切换 `p0` 时，categorical LUT 输出类别改变 15.39%；
- 切换 `p1` 时，categorical LUT 输出类别改变 15.36%；
- real/imag 单输出 bit 分别改变约 8.54%–8.57%。

因此模型并未忽略第三 bit。它学习出了明显不同于 LUT4 base 的 dominance-conditioned
逻辑，而且这种分化带来了真实 accuracy 收益。该结果同时说明，之前失败的关键不是
dominance 信息无效，而是 64-entry LUT6 无约束独立优化导致的稀疏、噪声化分裂。

### 当前方法的准确描述

- 每个二值复数 activation 原本由 `(sr,si)` 两 bit 表示。
- 两个复数 activation 一组，因此基础地址是四 bit，进入 LUT4。
- 每个复数额外增加一个 dominance bit `p`；一组总共增加 `p0,p1` 两 bit，
  地址从 LUT4 扩展成 LUT6。
- LUT6 logits 不独立随机学习，而由冻结 LUT4 base 加零均值 dominance residual 构成。
- alpha 从 0.1 退火到 1.0，residual LR=0.002，普通网络 continuation LR=0.0002。
- dominance bit forward 为硬比较器，向前 activation 的梯度被截断。
- 最终 checkpoint 的 alpha=1.0，forward 是完全 hard、可物化的普通 LUT6。

### 结论

这是当前首个证明额外 dominance bit 能带来正向精度收益的 LUT6 实验。关键贡献来自
“保留已学习 LUT4 operation，再受控学习条件 residual”，而不是单纯增加 LUT entry
数量。后续分析应以该 checkpoint 作为 dominance LUT6 基线。

### 文件修改总结

- `EXPERIMENT_LOG.md`：记录本次完整结果、里程碑、LUT sensitivity 和路线结论。
<!-- experiment-entry:dominance-shared-residual-layer-analysis-20260819 -->
## 2026-08-19 - 84.28% residual checkpoint 逐层利用率分析

通过 `scripts/analyze_dominance_lut_checkpoint.py` 对 best checkpoint 做逐层分析。

### 观察

- stage2 的 dominance category sensitivity 从 6.3% 逐步增加到约 10.6%。
- stage3 约为 10.5%–15.4%，末层明显更高。
- stage4 约为 14.5%–19.2%，最高为 `stage4.1` 的约 19.2%。
- p0/p1 sensitivity 在每层都高度接近，未观察到某一个额外 bit 被系统性忽略。
- centered residual RMS 从浅层约 0.36–0.45 增长到深层约 0.50–0.63。
- categorical top1-top2 mean margin 约 2.9–3.6，硬表整体并非处于普遍临界翻转状态。

### 推断

dominance/phase 信息在深层更有价值；统一 residual LR 和统一 alpha 会让低收益浅层承担
不必要的 LUT 分化噪声。下一步优先级：

1. 从 epoch194 best checkpoint 以 alpha=1、较小 continuation LR 延长训练，确认当前
   200 epochs 是否尚未收敛。
2. 增加 stage-wise residual gate/LR：stage2 更小或冻结，stage3 中等，stage4 保持完整。
3. 将四个 `(p0,p1)` slice 改为 Walsh/main-effect residual 参数化，优先学习 p0、p1
   主效应，再以较小尺度学习 p0×p1 interaction；最终仍物化为同一个 LUT6。
4. 在改变 phase-bit 公式前，先统计真实训练数据的 p0/p1 和 64-address occupancy，
   排除稀有 slice 导致的样本不足。

### 文件修改总结

- `EXPERIMENT_LOG.md`：记录逐层 sensitivity、residual 强度与后续实验优先级。
<!-- experiment-entry:continuation-and-occupancy-analysis-20260820 -->
## 2026-08-20 - residual 续训结果与真实地址 occupancy

### 续训结果

使用 `scripts/analyze_training_run.py` 分析：

`runs/phase3_lut6_dominance_sharedres_continue_lr1e4`

- 完成 100/100 epochs，所有指标 finite。
- best test accuracy：84.56%，续训 epoch 77。
- final test accuracy：84.21%。
- 相对上一阶段 84.28% 再提升 0.28 pp。
- best checkpoint 为 alpha=1.0 的完全 hard LUT6。
- p0/p1 静态 category sensitivity 分别为 17.04%/17.02%，单输出 bit sensitivity
  约 9.5%。

### 新增可复用分析脚本

新增 `scripts/analyze_dominance_occupancy.py`。脚本从 checkpoint 恢复实际模型，
在真实数据前向中通过 dominance LUT6 pre-hook 逐层、逐 group 累积地址，报告：

- p0/p1 的 0/1 分布；
- (p0,p1) 四组合分布；
- 每个 group 的 64-address coverage、零命中、低命中比例；
- normalized address entropy；
- LUT category sensitivity；
- 按真实地址命中次数加权的 sensitivity；
- 已分化但未访问/低访问的 LUT entry 比例；
- 最高与最低频地址。

保存报告：

- `reports/dominance_occupancy_continue_best_train10000.json`
- `reports/dominance_occupancy_continue_best_test10000.json`

### 10,000 个训练样本结果

共记录 5,164,160,000 次 LUT group lookup。

- p0=1：48.25%；p1=1：47.83%，边际分布接近均衡。
- (p0,p1)：00=31.70%，01=20.05%，10=20.47%，11=27.78%。
- 两个 bit 存在正相关，但四种组合均有充足样本，不构成严重 collapse。
- 所有层/group/address 合计零命中率仅 0.835%。
- hit-weighted category sensitivity：18.73%，高于静态平均水平，说明学到的
  phase-conditioned 变化倾向于落在常访问地址上。

稀疏性主要集中在最浅层：

- stage2.0：22.44% group-address 未访问，30.63% 少于 100 hits；
  8.28% 的 sensitive entries 未访问，17.61% 少于 100 hits。
- stage2.1：8.59% 未访问，11.53% 少于 100 hits；
  1.11% sensitive entries 未访问。
- stage2.2/2.3：未访问约 0.31%/0.94%。
- 从 stage2.4 开始，中深层几乎全部覆盖 64 地址。
- stage3/4 normalized entropy 多为 0.84–0.91，地址利用充分。
- 深层 hit-weighted sensitivity 约 23%–34%，明显高于浅层。

完整 test set 与训练增强样本结论一致：p0/p1 仍接近均衡，整体 group-address
零命中率为 0.908%，不存在 train/test occupancy 失配。

### 结论和下一步

第四步检查排除了“全网地址严重稀疏”和“第三 bit 单边 collapse”。当前 residual
已经把变化集中到高频地址，主要可优化点仅在 stage2.0/2.1：

1. 不应做全网 occupancy balancing，也不应继续强行提高 LUT 翻转率。
2. stage2.0/2.1 使用较小 residual alpha/LR，或采用共享主效应参数化，减少低频
   group-address 的独立噪声。
3. stage2.2 之后保持当前 full residual，尤其 stage3/4 不应收缩。
4. 对浅层低命中 entry 可使用 occupancy-weighted residual regularization，使其回归
   LUT4 base；最终硬件表不变。
5. Walsh 参数化优先在 stage2.0/2.1 试验：先学习 p0/p1 主效应，再决定是否开放
   p0*p1 interaction。

### 文件修改总结

- `scripts/analyze_dominance_occupancy.py`：新增真实数据逐层/逐 group occupancy
  和 hit-weighted LUT sensitivity 分析。
- `reports/dominance_occupancy_continue_best_train10000.json`：保存训练样本报告。
- `reports/dominance_occupancy_continue_best_test10000.json`：保存完整测试集报告。
- `EXPERIMENT_LOG.md`：记录续训结果、occupancy 指标和后续决策。
### 补充：浅层两路 activation 的联合相关性

stage2.0/2.1 的最高频地址集中在 `0,9,18,27,36,45,54,63`。按照
`[sr0,si0,p0,sr1,si1,p1]` 解码，这些地址满足第一个三 bit code 与第二个
三 bit code 相同。说明当前同 channel 空间位置配对在浅层具有明显相关性，两路输入
频繁表达同一复数状态，因此 LUT6 的联合地址熵较低。这与 stage2.0 normalized
entropy 约 0.64、深层约 0.84–0.91 的结果一致。

除了收缩浅层 residual，后续还可只针对 stage2 测试更低相关性的配对，例如拉开空间
tap 距离或跨 channel 配对；stage3/4 保持当前布局。该实验会改变路由代价，必须同时
评估 FPGA wiring，不应直接作为默认方案。

### 文件修改总结

- `EXPERIMENT_LOG.md`：补充最高频地址的 code-equality 解释和浅层配对优化方向。
## 2026-08-20：dominance LUT6 的 Walsh residual 参数化

### 目标

在不改变 LUT6 硬件接口和最终 64-entry 真值表的前提下，将原先四个
`(p0,p1)` slice 独立学习再减均值的 residual，改写为 Walsh-Hadamard 基底。
希望通过跨 slice 共享主效应，提高浅层地址 occupancy 较稀疏时的统计效率。

令 `q0=2*p0-1`、`q1=2*p1-1`，对每个旧 LUT4 地址和四个输出类别维护：

```
R(p0,p1) = q0*A + q1*B + q0*q1*I
logits6 = logits4_base + alpha*R
```

其中 `A` 是第一个 dominance bit 的主效应，`B` 是第二个 dominance bit
的主效应，`I` 是两者的交互效应。四个 slice 的 residual 均值天然为零。
`A/B/I` 共三个自由度，与原四 slice 减去均值后的三个有效自由度完全相同，
因此这是无损换坐标，不是表达能力压缩。

### 实现与兼容性

新增命令行参数值：

```
--pair-lut-parameterization categorical_walsh
```

要求仍为 `--pair-lut-inputs 6 --pair-lut-encoding dominance`。LUT4 checkpoint
warm start 时，冻结的 `dominance_base` 读取原 `16x4` categorical logits，
`A/B/I` 全部初始化为零，所以初始模型与 LUT4 逐地址完全等价。训练继续沿用
`dominance_residual_lr` 和 `dominance_residual_alpha` ramp。前向最终仍展开成
普通的 `64x2` hard LUT bits 并调用现有 LUT6 kernel，不增加 FPGA LUT、引脚或
输出资源。

### 文件修改总结

- `complexPyTorch/complexLayers.py`：新增 `categorical_walsh`；参数形状为
  `[Cout, group, 16, 3, 4]`，注册 `[q0,q1,q0*q1]` 基底，并在 categorical
  logits 生成阶段展开为 64-entry LUT6。
- `complexPyTorch/complexBinaryResNet.py`：模型参数校验允许 dominance LUT6
  使用 `categorical_walsh`。
- `training.py`：命令行、LUT4 warm start、checkpoint 恢复、独立 residual
  optimizer group 和 alpha ramp 接入 Walsh 参数化。
- `run_phase3.sh`：Walsh 模式下显示 residual LR 与 alpha ramp 配置。
- `tests/test_phase12_route.py`：新增 LUT4→Walsh 整网等价测试，以及任意零均值
  四 slice residual 可被 Walsh `A/B/I` 无损重建的测试。
- `scripts/analyze_dominance_lut_checkpoint.py`：支持读取 Walsh checkpoint，
  展开有效 LUT6 logits，并报告 `p0/p1/interaction` coefficient RMS。
- `EXPERIMENT_LOG.md`：记录本次公式、实现、硬件兼容性和验证结果。

### 验证

- 核心 Python 文件和分析脚本通过 `py_compile`。
- `run_phase3.sh` 通过 `bash -n`。
- Walsh 三项针对性测试通过。
- 完整 `tests.test_phase12_route` 回归测试：`52/52 passed`。
- 临时 Walsh checkpoint 可由分析脚本正确加载和展开。

## 2026-08-20：Walsh residual 首轮结果分析（运行中快照）

### 统一预处理

先使用 `scripts/analyze_training_run.py` 对 Walsh、旧 shared residual 主训练和续训
生成统一 JSON，再使用 `scripts/analyze_dominance_lut_checkpoint.py` 分析
best/last checkpoint 的 hard category sensitivity 与 residual 强度。本次观察时
Walsh 任务仍在运行，主进程为 PID 34127，约进行到 epoch 126，因此以下是运行中快照。

### 结果

| 实验 | weight/BN LR | residual LR | 最佳 test acc | 最佳 epoch |
|---|---:|---:|---:|---:|
| LUT4 warm-start 源模型 | 0.02（原始完整训练） | - | 82.53% | 248 |
| Walsh 首轮 | 0.02 multistep | 0.002 | 79.09% | 1 |
| shared residual 主训练 | 0.0002 constant | 0.002 | 84.28% | 194 |
| shared residual 续训 | 0.0001 constant | 0.001 | 84.56% | 77 |

Walsh 从 LUT4 checkpoint warm start 在数学和单测上完全等价，因此训练前应保持源模型
82.53% 的精度。但第一次 epoch 更新后立即降到 79.09%，随后到约 epoch 126 只有
约 77.5%–77.8%。此次命令将已收敛共享参数的 LR 设为 0.02，比成功的 residual
主训练 0.0002 大 100 倍，比续训 0.0001 大 200 倍。旧 LUT4 的 0.02 是从较差起点
完成整段训练时使用的 LR，不能直接用于已经处于局部最优点的 warm-start 续训。

### LUT 是否学动

Walsh best checkpoint（epoch 1，alpha=0.1）：

- p0/p1 category sensitivity 均约 0.36%；
- 此时 LUT6 基本仍等价于 LUT4。

Walsh 当前 last checkpoint（约 epoch 126，alpha=1）：

- p0 category sensitivity 约 16.19%；
- p1 category sensitivity 约 16.19%；
- `A`、`B`、`I` 的逐层 RMS 均值分别约 0.3330、0.3334、0.3119；
- 展开后的 residual RMS 均值约 0.5653。

旧 shared residual 最佳 checkpoint：

- p0/p1 category sensitivity 约 17.04%/17.02%；
- centered residual RMS 均值约 0.5277。

因此 Walsh 并没有再次出现 LUT 不翻转的问题。它已经学出和旧 residual 相近数量的
hard table 差异；精度差主要来自共享 weight/BN 在第一轮被过大 LR 破坏。

### Walsh LR 尺度

对一个已命中的 dominance slice，旧参数化
`z_s-mean(z)` 的参数梯度平方范数为 `3/4`；Walsh
`q0*A+q1*B+q0*q1*I` 的参数梯度平方范数为 `3`。同一 residual LR 下，
Walsh 的局部函数空间更新尺度约为旧参数化的 4 倍。Adam 会削弱但不会完全消除这种
差异。因此下一次干净实验建议：

- weight/BN：`LR=0.0002`、`SCHEDULE=constant`；
- Walsh residual：先用 `DOMINANCE_RESIDUAL_LR=0.0005`；
- alpha 仍从 0.1 在 120 epochs 内升到 1；
- 从同一个 LUT4 best checkpoint 重新开始，使用新 WORKDIR；
- 当前 `LR=0.02` 的 run 已无继续判断 Walsh 优劣的价值，可停止。

### 文件修改总结

- `reports/phase3_lut6_dominance_walsh_lr002.json`：Walsh 训练统一报告。
- `reports/phase3_lut6_dominance_sharedres_continue_lr1e4.json`：旧 residual
  续训统一对照报告。
- `reports/phase3_lut6_dominance_sharedres_lr2e4_reslr2e3.json`：旧 residual
  主训练统一对照报告。
- `reports/phase3_pair_lut4_categorical_phase2init_sf11_lr002.json`：LUT4
  warm-start 源模型报告。
- `reports/phase3_lut6_dominance_walsh_best_lut.json`：Walsh best LUT 指标。
- `reports/phase3_lut6_dominance_walsh_last_lut.json`：Walsh last 汇总指标。
- `reports/phase3_lut6_dominance_walsh_last_lut_full.json`：Walsh last 逐层指标。
- `reports/phase3_lut6_dominance_sharedres_continue_best_lut.json`：旧 residual
  最佳 checkpoint 的逐层 LUT 对照指标。
- `EXPERIMENT_LOG.md`：记录本次配置错误、LUT 指标和下一实验决策。

## 2026-08-21：clean Walsh 完整结果与参数化结论

### 结果

使用修正后的公平配置完成 200 epochs：

- weight/BN LR：0.0002 constant；
- Walsh residual LR：0.0005；
- alpha：0.1→1.0，120 epochs；
- LUT4 best checkpoint warm start；
- dominance bit stop-gradient。

最终结果：

- best test acc：83.53%（epoch 176）；
- final test acc：83.25%；
- LUT4 warm-start 源模型：82.53%；
- direct shared residual 主训练：84.28%；
- direct shared residual 续训最佳：84.56%。

因此 clean Walsh 相比 LUT4 确实获得约 1.00 个百分点收益，但相比直接维护四个
dominance slice 的 residual 仍低约 1.03 个百分点。此前 LR=0.02 的失败不能用于
判断 Walsh；本次 clean run 才是有效比较。

### LUT 指标

Walsh best checkpoint：

- p0 category sensitivity：7.29%；
- p1 category sensitivity：7.31%；
- A/B/I RMS 均值：0.1393 / 0.1392 / 0.1189；
- interaction 与两个 main-effect 平均 RMS 之比：85.35%；
- 展开后的 residual RMS：0.2303。

旧 direct residual 最佳 checkpoint：

- p0/p1 category sensitivity：17.04% / 17.02%；
- centered residual RMS：0.5277；
- best test acc：84.56%。

交互项强度接近主效应，说明最优 LUT6 操作明显依赖两个复数输入及其 dominance bit
的联合状态，不是主要由可分离的 p0、p1 factor 构成。

### 原因分析

Walsh 与 direct zero-mean residual 在数学表达能力上等价，均有三个有效自由度；
性能差异来自优化坐标而非硬件表达能力：

- LUT6 地址本身已经联合包含
  `[sr0,si0,p0,sr1,si1,p1]`，直接 residual 对每个 6-bit 地址局部更新；
- Walsh 中更新 A 会同时改变两组 p0 slice，更新 B 会同时改变两组 p1 slice，
  更新 I 会以 checkerboard 方式同时改变四个 slice；
- 某一个具体地址只需要翻转一个输出类别时，Walsh 必须协调 A/B/I 才能保持另外三个
  slice 不变；
- categorical hard argmax 使这种联动更容易产生无关 slice 的 margin 扰动；
- Adam 不具备对这种线性坐标变换的旋转不变性，因此即使函数空间等价，优化轨迹也
  不等价。

这与“LUT6 已经蕴含两个复数的联合 factor，外部再次结构化会增加优化难度”的判断
一致。Walsh 在这里不是硬件上的外部 factor，但确实是训练参数上的额外结构约束。

### 路线决策

不建议继续把 Walsh 作为主路线。恢复 direct `categorical_residual`：

- 它与 LUT6 的 address-local 语义最一致；
- 当前已有 84.56% 最佳结果；
- hard sensitivity 更高，说明第五、第六 bit 被更充分利用；
- 最终硬件资源与 Walsh 完全相同。

Walsh 代码可作为消融实验保留，不设为默认，不继续投入主要实验额度。

### 文件修改总结

- `reports/phase3_lut6_dominance_walsh_clean_lr2e4_reslr5e4.json`：clean Walsh
  完整训练报告。
- `reports/phase3_lut6_dominance_walsh_clean_best_lut.json`：clean Walsh best
  checkpoint 的逐层 A/B/I、margin 和 sensitivity 报告。
- `EXPERIMENT_LOG.md`：记录 clean Walsh 最终结果、优化机理和路线决策。

## 2026-08-21：categorical residual 后续 idea 筛选

基于当前累计结果（LUT4 82.53%、direct categorical residual 84.56%、clean Walsh
83.53%）逐项评估新的改进建议。

### 方向 1：hierarchical residual

原提案
`B + alpha1*D_phase + alpha2*D_corr` 若两个 residual 都是完整的 64-address
四分类 logits，则存在不可辨识性：同一修正可以任意分配给两个张量，只增加优化冗余。
若将 `D_phase` 限制为 p0/p1 主效应、`D_corr` 表示交互或 full correction，
本质上又回到 Walsh/ANOVA staged release。clean Walsh 已证明这种跨 slice 联动
不如 address-local residual。因此不建议按原形式实现。

### 方向 2：complex/factorized residual

将 residual 拆成独立 real/imag logits 会取消当前成功的四分类联合约束。最终硬件虽是
两张 LUT，但当前训练通过同一个四分类状态联合决定实虚输出；拆分后会重新出现两张表
各自优化、复数输出组合缺乏约束的问题。可作为低优先级消融，不作为主路线。

### 方向 3/7：替换 third bit

magnitude bit、learnable threshold 和带梯度的第五 bit 在历史路线中已多次出现训练
下降或局部最优问题。当前 dominance bit 已经带来约 2.03 个百分点收益，不应立即替换。
若重启该方向，应先用现有 occupancy 脚本比较候选 bit 的 balance、地址 entropy 和
与 hard LUT correction 的互信息，再决定是否做完整训练。learnable gamma 还会引入
额外硬件尺度比较代价，除非量化成移位可实现的有限集合。

### 方向 4：phase interaction 分解

`D0(p0)+D1(p1)+D01(p0,p1)` 就是已经完成的 Walsh/ANOVA 参数化。其表达空间与
direct zero-mean residual 等价，但 clean 结果只有 83.53%，低于 direct residual
84.56%。该方向已经完成验证，不再重复。

### 方向 5：residual regularization

普通全局 L2 可用，但会对高频、确有收益的 correction 一并施压，并且 raw categorical
logits 存在共同平移不改变 softmax 的 gauge。更推荐 occupancy-aware functional
regularization：

- 对同一 LUT4 base 下的 centered residual 或 categorical KL 施加约束；
- 低 occupancy 地址使用更大 lambda，使其回归 base；
- 高频地址使用较小 lambda，保留第五/第六 bit 的有效修正；
- lambda 随 epoch 逐渐减小或在 alpha ramp 结束后固定到很小值。

这直接针对之前发现的 stage2.0/2.1 稀疏地址问题，且不改变硬件。

### 方向 6：knowledge distillation

有潜力，但不建议使用 `||Z6-Z4||^2` 的 raw-logit MSE。更合理的是：

`KL(softmax(B/T) || softmax((B+alpha*R)/T))`

并采用 early-only、逐渐衰减、occupancy-aware 的权重。它相当于 categorical
trust region：训练初期防止低置信地址无依据翻转，后期释放 student 超过 teacher。
由于 alpha ramp 已具有类似保守作用，distillation 应先作为小规模消融，避免过强约束
把模型锁回 82.53% teacher。

### 推荐优先级

1. direct categorical residual + occupancy-aware categorical KL/L2；
2. 更简单的 layer-wise residual LR：stage2.0/2.1 降低，stage3/4 保持当前值；
3. early decaying categorical KL distillation；
4. 在统计信息证明更好后，再测试固定 third-bit 替代方案；
5. real/imag independent residual 仅作消融；
6. 不再测试 Walsh/ANOVA 或冗余 hierarchical full residual。

### 文件修改总结

- `EXPERIMENT_LOG.md`：记录 idea 与历史实验的对应关系、淘汰理由和下一实验优先级。

## 2026-08-21：当前 categorical residual 执行顺序审计

按当前代码重新核对 LUT4 checkpoint warm start、activation、LUT6 地址生成、
categorical residual、CUDA 查表、BN/residual block 和反向传播。

关键事实：

1. Phase3 block 顺序为 pre-activation：保存原 inp 作为 identity 和
   dominance_source，BinaryComplexBitActivation 将实虚符号编码为 0/1，LUT 查表累加，
   covariance complex BN，最后与 identity/projection 相加。
2. dominance 实际公式是
   `delta=sign(i)*r-sign(r)*i=sign(r)sign(i)(|r|-|i|)`，
   `p=1[delta>=0]`。它是象限相关的 phase/Gray bit，并非始终等于
   `1[|r|>=|i|]`。当前 stop 模式直接 detach，不经 p 向前层传梯度。
3. LUT6 地址顺序严格为 `[r0,i0,p0,r1,i1,p1]`，整数地址为
   `32r0+16i0+8p0+4r1+2i1+p1`。旧 LUT4 地址为
   `8r0+4i0+2r1+i1`。
4. LUT4 categorical logits 作为冻结 buffer `dominance_base[O,G,16,4]`；
   可学习 residual 为 `weight[O,G,64,4]`，初始化全零。对同一旧地址的四个
   `(p0,p1)` slice 按 class 分别减均值，再计算
   `base+alpha*centered_residual`。
5. 每个地址经四分类 softmax + hard argmax STE 联合选择
   `00/01/10/11` 复数输出状态，再展开成两张 `[O,G,64]` 的 0/1 LUT。
6. CUDA 前向将六个输入 bit 打包为 0–63 地址，读取 hard table；所有 group 的
   0/1 输出直接求和，所以 BN 前实部、虚部范围均为 `[0,G]`。post complex BN
   负责去直流、协方差白化和可学习复仿射变换。
7. CUDA 对 LUT table 的梯度只累积到命中地址；输入 bit 梯度采用
   `table[idx(bit=1)]-table[idx(bit=0)]` 的有限差分。real/imag bit 梯度继续经过
   Bi-Real surrogate；p 因 stop-gradient 不继续传播。
8. residual raw 参数梯度还会经过四 slice 零均值投影，并乘当前 alpha。
   base 是 buffer，不更新。训练全程 hard，没有 tau/soft-forward，只有 alpha ramp。
9. 当前 `F.unfold` 的 logical position 顺序是 channel-major、channel 内 kernel
   row-major，然后每两个位置一组。由于 3x3 每 channel 有 9 个 tap，边界 pair 会
   跨 channel：例如 `(c0,k22)` 与 `(c1,k00)`。若总 logical position 为奇数，
   最后一个位置会与整个 patch 的第一个位置配对。当前实现因此是“多数 pair 同 channel”，
   不是“所有 pair 严格同 channel”。

### 文件修改总结

- `EXPERIMENT_LOG.md`：记录当前 categorical residual 的实际代码顺序、梯度路径、
  地址公式和 channel-major pairing 边界行为。


## 2026-08-21：固化 84.56% categorical residual 主路线并移除 Walsh

### 结果与决策

当前最佳模型来自 `categorical_residual` dominance LUT6：LUT4 categorical checkpoint
先扩展为完全等价的 LUT6 base，再只学习每个 64-address 的零均值四分类 residual。
主训练最佳 84.28%，续训最佳 **84.56%**（epoch 77），相对 LUT4 warm-start 的
82.53% 提升 2.03 个百分点。p0/p1 hard category sensitivity 为 17.04% /
17.02%，说明两个新增 phase/Gray bit 已实际改变硬真值表。

clean Walsh 对照仅达到 83.53%。虽然其函数空间与零均值 direct residual 等价，
但跨 dominance slice 的联动坐标增加了 hard categorical 优化难度。因此本次从模型、
CLI、调度、测试和 checkpoint 分析脚本中完全移除 Walsh，不再提供该运行选项；历史
实验结论保留在本日志中，避免未来重复测试。

### 验证

- `python -m py_compile`：核心模型、训练入口和 dominance checkpoint 分析脚本通过；
- `bash -n run_phase3.sh`：通过；
- `python -m unittest tests.test_phase12_route`：50 tests passed；
- 保留 LUT4->LUT6 等价 warm start、zero-mean residual、alpha/LR schedule 测试。

### 文件修改总结

- `complexPyTorch/complexLayers.py`：清理历史行尾空格；删除 Walsh 参数张量、basis 展开和分支，只保留
  independent、categorical 与当前最佳 categorical_residual。
- `complexPyTorch/complexBinaryResNet.py`：删除 Walsh 模型配置入口。
- `training.py`：删除 Walsh checkpoint、optimizer、flow scheduler 和 CLI 分支；修正
  dominance bit 的帮助文本为 phase/Gray comparator bit。
- `run_phase3.sh`：删除 Walsh residual 调度分支。
- `tests/test_phase12_route.py`：删除 Walsh 专用测试，保留主路线回归覆盖。
- `scripts/analyze_dominance_lut_checkpoint.py`：删除 Walsh coefficient 分析，仅分析当前
  64x4 direct categorical residual checkpoint。
- `CURRENT_TECHNICAL_ROUTE.md`：记录 84.56% 最佳结果、续训命令和硬件映射结论。
- `EXPERIMENT_LOG.md`：记录本次路线固化、验证结果及所有文件修改。

### 本次路线快照中一并固化的历史文件变化

- `lut_cuda/lut_conv_cuda_backend.cu`：保留此前已验证的 inactive lane 非有限梯度修复；
  本次未再修改 CUDA kernel。
- `run_phase2.sh`、`scripts/analyze_training_run.py`、`scripts/audit_active_route.py`：固化
  当前 Phase1/2/3 路线的启动与可复用审计工具。
- `scripts/analyze_checkpoint_nonfinite.py`、`scripts/analyze_nonfinite_gradient_report.py`、
  `scripts/analyze_lut_reference_gap.py`、`scripts/analyze_dominance_occupancy.py`：纳入已用于
  实验分析的可复用脚本。
- 删除旧 `run_phase2_complex_bireal.sh`、`run_phase3_pair_analytic.sh`、
  `run_phase3_real_compatible.sh`、旧 pair-LUT 分析脚本、旧独立测试文件和未使用的
  `lut_conv_bafw_cuda_backend_top1.cu`，使活动目录只呈现当前 Phase1/2/3 路线。

## 2026-08-21：Git 工作目录整理

本次只整理版本控制边界，不删除本地 checkpoint、数据集、backup 或生成报告。历史上
误跟踪的 Python bytecode、CUDA shared objects、object files 与 nested build 目录已从
Git index 移除，并由 ignore 规则统一管理。由失败补丁工具产生的 `.orig` 临时副本和
两个空占位文件已从本地删除。

### 文件修改总结

- `.gitignore`：忽略 native build 产物、backup、checkpoint 工作目录、数据集、报告、
  外部参考代码、PDF 与补丁临时文件。
- `EXPERIMENT_LOG.md`：记录本次 Git 目录整理的范围与不删除本地资产的约束。
- `scripts/analyze_dominance_lut_checkpoint.py`：去掉文件尾多余空行，仅做格式整理。
- `complexPyTorch/__pycache__/`、`lut_cuda/build/`、`lut_cuda/*.so`：仅取消 Git 跟踪，
  本地构建文件仍保留，后续可正常运行和重编译。

## 2026-08-21：加入可控 LUT4 common correction

### 动机与公式

当前最佳公式只包含冻结 LUT4 base 与四个 dominance slice 上零均值的 LUT6 residual，
因此无法直接表达四个 slice 共同需要的 LUT4 operation 修正。本次加入
`dominance_base_correction[O,G,16,4]`：

`logits = frozen_base + beta * class_center(common) + alpha * slice_center(residual)`。

common correction 在类别维减均值，residual 在同一旧 LUT4 地址的四个 dominance
slice 上减均值，因此两部分正交且可解释。训练结束仍只导出两张 hard LUT6，硬件资源
不变。

### 控制与兼容性

- 新增独立 `dominance_base` optimizer group；
- 新增 `DOMINANCE_BASE_LR`、`DOMINANCE_BASE_ALPHA_START/END` 和
  `DOMINANCE_BASE_RAMP_EPOCHS`；
- 默认 base LR 与 alpha 均为 0，旧命令行为不变；
- 旧 84.56% checkpoint 缺少 correction 时强制零初始化，初始 hard forward 不变；
- checkpoint 分析脚本新增 base alpha 与 centered correction RMS。

### 推荐首轮配置

从 84.56% best checkpoint 续训 100 epochs：普通参数 LR `5e-5`，已有 residual LR
`5e-4`，common correction LR `1e-4`；residual alpha 固定 1，base alpha 在前 20
epochs 从 0 增至 1。完整命令记录在 `CURRENT_TECHNICAL_ROUTE.md`。

### 验证

- 核心 Python 文件编译检查通过；
- `run_phase3.sh` shell 语法通过；
- `tests.test_phase12_route`：53 tests passed；
- 新测试覆盖 LUT4 warm start 零 correction、common correction 的 slice 共享与类别
  零均值、独立 LR/alpha schedule，以及旧 residual checkpoint 加载前后输出等价。

### 文件修改总结

- `complexPyTorch/complexLayers.py`：新增 class-centered LUT4 common correction、beta
  buffer 和 setter，并合入 categorical logits。
- `training.py`：新增旧 checkpoint 兼容、独立 optimizer group、base alpha schedule、
  参数校验、训练日志与 checkpoint metrics。
- `run_phase3.sh`：暴露 common correction 的 LR 和 alpha 环境变量并打印配置。
- `tests/test_phase12_route.py`：新增 common correction、调度和 legacy checkpoint 测试。
- `scripts/analyze_dominance_lut_checkpoint.py`：分析 effective base correction 及其 RMS。
- `CURRENT_TECHNICAL_ROUTE.md`：记录新公式、默认兼容行为和推荐续训命令。
- `EXPERIMENT_LOG.md`：记录本次设计、验证结果和全部文件修改。

## 2026-08-21：增加同配置 from-scratch 初始化消融

为判断 84.56% checkpoint 是否把新模型限制在旧 operation 的局部最优，新增不读取
checkpoint、其余超参数完全一致的并行实验。检查发现原 `categorical_residual` 构造函数
将 base 与 residual 全部置零，会导致所有 hard LUT entry 初始选择同一类别，因此不能
作为有效 scratch 对照。

现在无 checkpoint 时：

- `dominance_base[O,G,16,4]` 使用 `N(0,0.01)`；
- `weight[O,G,64,4]` residual 使用 `N(0,0.01)`；
- `dominance_base_correction` 仍为 0；
- hard categorical table 从 epoch 1 即具有多个随机输出类别；
- LUT4 warm start 仍复制 checkpoint base，并清零 residual/correction，行为不变。

严格同配置 scratch 组继续使用普通参数 LR `5e-5`、residual LR `5e-4`、common LR
`1e-4`。该实验只改变初始化；若失败，只能说明低 LR continuation schedule 依赖预训练
起点，不能单独证明随机 LUT6 无法训练。完整 GPU3 命令写入
`CURRENT_TECHNICAL_ROUTE.md`。

### 文件修改总结

- `complexPyTorch/complexLayers.py`：将无 checkpoint 的 categorical residual base 与
  residual 改为小随机初始化；checkpoint 路径仍显式覆盖并清零。
- `tests/test_phase12_route.py`：新增 scratch hard-category 破除对称测试，并复用 warm-start
  测试确认 checkpoint 起点不受影响。
- `CURRENT_TECHNICAL_ROUTE.md`：记录 GPU3 from-scratch 单变量消融命令和结论边界。
- `EXPERIMENT_LOG.md`：记录初始化问题、修复、实验设计和文件修改。

## 2026-08-21：common correction 结果与 LUT4 起点联合释放消融

使用 `scripts/analyze_training_run.py` 统一处理两组新日志并生成报告。

### 已完成结果

`runs/phase3_lut6_commonbase_lr1e4_reslr5e4`：

- 状态：完成 100/100 epochs，全部 finite；
- 起点：已训练好的 84.56% LUT6 categorical residual checkpoint；
- best test：85.09%，epoch 39；
- final test：84.34%；
- 相对 84.56% 提升 0.53 pp，相对 LUT4 82.53% 提升 2.56 pp；
- 略高于 Phase2 85.05% 约 0.04 pp；
- 后 61 epochs 回落 0.75 pp，说明 common correction 有收益但继续训练会产生 drift。

`runs/phase3_lut6_commonbase_fromscratch_samecfg`：

- 状态：用户在 epoch 60/100 停止；
- best test：57.76%，epoch 57；
- final test：57.04%；
- 该结果验证普通参数 LR `5e-5` 的 continuation 配置不能从随机网络恢复，不支持
  “完全 scratch 更容易跳出局部最优”的假设。

结构化报告：

- `reports/phase3_lut6_commonbase_lr1e4_reslr5e4.json`；
- `reports/phase3_lut6_commonbase_fromscratch_samecfg.json`。

### 新消融：从 LUT4 同时释放两部分

此前 85.09% 实验的 LUT6 residual 已经预先训练到 84.56%，随后才释放 LUT4 common
correction，属于 sequential optimization。新的 joint 实验直接读取 82.53% categorical
LUT4 checkpoint：

- `dominance_base` 精确复制 LUT4 logits 并冻结；
- LUT6 zero-mean residual 初始化为 0；
- LUT4 class-centered common correction 初始化为 0；
- residual 与 common alpha 均 `0.1 -> 1.0 / 120 epochs`；
- residual LR `0.002`，common LR `0.0002`，共享 weight/BN LR `0.0002`；
- 总训练 200 epochs，最终硬件仍只有两张 hard LUT6。

这与原 84.28% residual 主训练只有一个差异：是否同时开放 LUT4 common correction，
因此是判断 sequential 与 joint release 的直接消融。完整命令写入
`CURRENT_TECHNICAL_ROUTE.md`。

### 文件修改总结

- `reports/phase3_lut6_commonbase_lr1e4_reslr5e4.json`：common continuation 统一报告。
- `reports/phase3_lut6_commonbase_fromscratch_samecfg.json`：停止的 scratch 统一报告。
- `CURRENT_TECHNICAL_ROUTE.md`：记录两组结果和 LUT4 起点 joint-release 命令。
- `EXPERIMENT_LOG.md`：记录定量结果、结论边界和新消融设计。

## 2026-08-22：LUT4/LUT6 joint release 完整结果与下一步

使用 `scripts/analyze_training_run.py` 和 `scripts/analyze_dominance_lut_checkpoint.py`
统一分析 joint 与 sequential best checkpoint。

### Joint 训练结果

`runs/phase3_lut4_joint_base_and_lut6_residual`：

- 完成 200/200 epochs，全部 finite；
- best test：84.61%，epoch 173；
- final test：83.89%；
- residual 与 common alpha 在 epoch 120 前后均已达到 1；
- full-release 后继续约 53 epochs 才达到 best，随后 27 epochs 回落 0.72 pp。

该结果高于 LUT4 82.53%，证明从 LUT4 起点同时开放 LUT6 residual 与 LUT4 common
correction 可以训练；但低于 sequential common-correction best 85.09%，而且 constant LR
后期继续产生 drift。差距不能简单归因于 joint 训练少了几十个 epoch，因为 joint 在 best
之后继续训练反而下降。

### Checkpoint 分解

| 指标 | joint best 84.61% | sequential best 85.09% |
|---|---:|---:|
| centered LUT6 residual RMS | 0.4498 | 0.5367 |
| centered LUT4 common correction RMS | 0.0522 | 0.0104 |
| p0 hard category sensitivity | 14.77% | 17.53% |
| p1 hard category sensitivity | 14.73% | 17.51% |

joint 的 common correction 强度约为 sequential 的 5 倍，同时 residual 强度和新增 bit
sensitivity 更低。逐层结果一致：joint common RMS 从浅层约 0.034 增至深层约
0.066–0.070；sequential 各层基本稳定在约 0.010–0.011。

因此虽然 common 与 residual 在参数分解上正交，hard argmax 和任务优化仍让两条路径
发生竞争：joint 中较早、较强的 common 更新吸收了本应由 dominance slice 分化表达的
改进，降低了第五、第六 bit 的利用。

### 下一步优先级

1. 将 85.09% best checkpoint 的 hard LUT 固定，短程只训练 BN、projection、stem 与
   classifier，检查是否能消除 LUT 翻转造成的后期 drift。
2. 从 LUT4 再做 constrained joint：residual 保持 LR `0.002`，common LR 从 `0.0002`
   降到约 `2e-5`–`5e-5`，或延迟到 residual alpha 接近 1 后再释放；目标是把 common
   RMS 控制在 sequential 观察到的约 0.01，而不是 0.05。
3. common correction 应被视为 trust-region 小修正，可增加独立 L2 penalty 或 norm
   clamp；不建议继续提高 common capacity/LR。
4. 后期不应继续使用恒定 LUT LR。alpha 达到 1 后，应降低或冻结 LUT LR，再给共享
   参数一个恢复阶段。

最优先建议是第 1 项，因为不增加硬件、不改变已有 85.09% hard LUT，而且直接针对两组
实验共同出现的 best 后回落。第 2 项用于验证更受限的 joint 是否能接近 sequential。

### 生成报告

- `reports/phase3_lut4_joint_base_and_lut6_residual.json`：joint 训练汇总；
- `reports/joint_commonbase_best_lut.json`：joint best 逐层 LUT 分解；
- `reports/sequential_commonbase_best_lut.json`：sequential best 逐层 LUT 分解。

### 文件修改总结

- `EXPERIMENT_LOG.md`：记录 joint 完整结果、checkpoint 分解、竞争机制与下一步优先级。

## 2026-08-22：85.09% 之后的其他无硬件开销优化方向

基于 LUT4 82.53%、LUT6 residual 84.56%、sequential common 85.09%、joint common
84.61% 以及 occupancy/逐层分解，筛选不增加最终 FPGA 资源的后续方案。淘汰项仍包括
Walsh、完全 scratch、强行提高 hard flip、继续增加 LUT capacity 和无约束 common LR。

### 1. Backward-only categorical temperature schedule

当前 forward 始终 hard argmax，backward 固定使用 `softmax(logits / tau)` 且 tau=1。
可以只调整代理梯度温度，forward 与硬件行为完全不变：初期 tau 约 1.5–2.0，使相邻
类别都获得梯度；后期降至约 0.5，使已形成 margin 的 hard category 稳定。该方案与历史
soft-forward annealing 不同，不会产生 soft/hard 前向精度落差。优先级高。

### 2. Occupancy-aware trust-region regularization

真实数据已经证明稀疏性集中在 stage2.0/2.1。对低命中地址施加更强的 categorical KL
或 centered-residual L2，使其回归 LUT4 base；高频地址保持自由。建议权重与
`1/sqrt(hit_count+1)` 或其归一化版本成正比，只作用于浅层。最终仍物化同一 hard LUT6。
这比全局 L2 更有针对性。

### 3. EMA/SWA 或 hard-table temporal voting

sequential best 85.09% 最终回落到 84.34%，joint best 84.61% 最终回落到 83.89%，
说明单 epoch hard table 存在时间漂移。训练期维护 logits/BN 参数 EMA，或对 plateau 多个
checkpoint 的每个 LUT entry 做 hard category 多数投票，再校准 BN，可输出唯一 hard table，
不增加推理资源。该方案直接利用已有训练轨迹，成本较低。

### 4. Phase2 teacher distillation

使用 85.05% Phase2 模型作为 teacher，对 Phase3 分类 logits 增加 early-decaying KL。
Teacher 不需要进入部署，只作为训练正则，帮助学生在探索 LUT6 operation 时保留旧网络
决策边界。更进一步可只蒸馏 stage2 浅层 block 输出，因为该处 occupancy 最稀疏。风险是
系数过大会把模型锁回 Phase2，因此应在 alpha ramp 完成前衰减到 0。

### 5. Stage-wise LUT learning rate

根据逐层统计，stage2.0/2.1 sensitivity 和 occupancy 最低，stage3/4 最高。可以给
stage2.0/2.1 使用 0.1x–0.25x residual/common LR，stage2.2 以后保持当前 LR。该方案
不改变参数化，只减少浅层稀疏地址噪声，优先级高于重新设计 third bit。

### 推荐实验顺序

1. 先做 hard LUT freeze + 非 LUT 参数恢复，验证 drift 是否可恢复；
2. 同时实现 backward-only tau schedule，比较固定 tau=1 与 `2.0 -> 0.5`；
3. 加入 EMA/hard-table voting，利用一个训练任务同时得到普通与 EMA checkpoint；
4. 再测试浅层 occupancy-aware regularization 或 stage-wise LR；
5. teacher distillation 作为后续独立方向。

其中 tau schedule、EMA 和 stage-wise LR 都只改变训练过程，最终硬件与当前 85.09%
模型完全相同。

### 文件修改总结

- `EXPERIMENT_LOG.md`：记录基于现有全部结果筛选出的无硬件开销优化方向及优先级。

## 2026-08-22：利用 Xilinx LUT6_2 的 O5/O6 增加局部输出信息

根据 AMD/Xilinx `LUT6_2` 原语文档，单个物理 LUT6 的 O5/O6 共享同一个 64-bit
INIT。令 `u=I[4:0]`、`s=I5`、真值表为 `T`，则：

- `O5(u)=T(0,u)`，O5 与 I5 无关；
- `O6(s,u)=T(s,u)`；
- 当 `s=0` 时，`O5=O6`；
- 当 `s=1` 时，O5 读取低 32 entries，O6 读取高 32 entries，可不同。

因此 O5/O6 不能实现两个任意 6-input Boolean function。它可以实现两个共享五个动态
输入的独立 LUT5（将 I5 固定为 1），或者同时暴露一个 LUT6 function 及其 I5=0
cofactor。官方来源：AMD UG953 `LUT6_2` 与 UltraScale UG574 LUT primitives。

### 与当前 dominance LUT6 的匹配

将某一个 dominance bit（建议 p1）接到 I5，其余 `[sr0,si0,p0,sr1,si1]` 接 I0-I4：

- O6 输出当前真实六位地址的 local bit；
- O5 同时输出“把 p1 强制为 0”时的 counterfactual/reference bit；
- p1=0 时两个输出重复；
- p1=1 时二者可以形成四种组合，显式保留 p1 是否改变 operation 的信息。

当前实部和虚部各使用一个物理 LUT6，因此两个 `LUT6_2` 可从原来的 2 个 bit
`[O6_r,O6_i]` 扩展为 4 根输出线 `[O6_r,O5_r,O6_i,O5_i]`，物理 operator LUT 数量
不增加，且 O5 不引入额外 INIT 参数。

### 不能忽略的下游成本

额外 O5 不是系统级零成本。如果分别累加四个 bit plane，会增加 accumulator、路由、
寄存器和 BN/后处理带宽。较紧凑的方案是对每个实/虚 LUT 输出形成二位局部码，例如
`q=2*O6+O5`，再作为 0..3 的 operand 累加；这不增加 operator LUT 数，但会让加法树
输入和累加器约增加 1 bit。实际 LUT/carry/route 成本必须通过综合报告判断。

### 推荐软件消融

第一版实现 `LUT6_2-aware 2-bit local output`：

1. 固定 p1 为 I5，保持现有 64-entry categorical LUT6 参数化；
2. O6 按实际 6-bit 地址查表；O5 按同一地址清除 p1 bit 后查同一张表；
3. 实部和虚部分别形成 `2*O6+O5`，再沿 group 维累加；
4. 后续仍进入现有 complex BN 和二值 activation；
5. 最终导出同一对 LUT6_2 INIT，不新增 LUT truth-table 参数；
6. 与只使用 O6 的 85.09% 基线比较精度，并用 Vivado 综合比较 accumulator/carry 成本。

另一个资源导向方案是把一个 LUT6_2 的 O6/O5 直接作为实部/虚部输出，从而把两个
物理 LUT 压成一个，但会强制其中一个输出等于另一个输出的 I5=0 cofactor，约束很强，
更适合作为面积消融而非当前精度优先主路线。

### 文件修改总结

- `EXPERIMENT_LOG.md`：记录 LUT6_2 O5/O6 的精确逻辑约束、当前网络映射、潜在收益、
  下游硬件代价及推荐消融。
